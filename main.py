"""Batch document conversion with LayoutParser-assisted Arabic PDF OCR.

For PDFs, every page is rendered at a high DPI, segmented with LayoutParser,
and each detected region is passed to MarkItDown's vision OCR independently.
This prevents text from different columns, tables, and figures being mixed into
one OCR request.  In addition to the Markdown file, the script writes:

* ``<document>.layout.json``: detected bounding boxes and OCR for each block.
* ``<document>_layout/page_XXXX.png``: page previews with detected boxes.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unicodedata
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

from markitdown import MarkItDown
from openai import OpenAI


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
OCR_MODEL = os.getenv("OCR_MODEL", "deepseek-ocr")

# 250 DPI is a good OCR baseline for Arabic diacritics and small text.  Increase
# it with PDF_OCR_DPI=300 when the original scan is particularly small/noisy.
PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "250"))
LAYOUT_SCORE_THRESHOLD = float(os.getenv("LAYOUT_SCORE_THRESHOLD", "0.25"))

# EfficientDet avoids requiring Detectron2, which is difficult to install on
# Windows. The default loader uses a safe local filename because LayoutParser
# 0.3.4's Windows cache cannot handle the ``?dl=1`` in its Dropbox URL.
DEFAULT_LAYOUT_MODEL_URI = "lp://efficientdet/PubLayNet/tf_efficientdet_d0"
LAYOUT_MODEL_URI = os.getenv("LAYOUT_MODEL_URI", DEFAULT_LAYOUT_MODEL_URI)
LAYOUT_MODEL_CACHE = Path(os.getenv("LAYOUT_MODEL_CACHE", ".layout_models"))
PUBLAYNET_WEIGHTS_URL = (
    "https://huggingface.co/layoutparser/efficientdet/resolve/main/"
    "PubLayNet/tf_efficientdet_d0/publaynet-tf_efficientdet_d0.pth.tar?download=true"
)
PUBLAYNET_WEIGHTS_NAME = "publaynet-tf_efficientdet_d0.pth.tar"
PUBLAYNET_LABEL_MAP = {
    1: "Text",
    2: "Title",
    3: "List",
    4: "Table",
    5: "Figure",
}

ARABIC_OCR_PROMPT = """<|grounding|>
convert this document region to Markdown.

"""

client = OpenAI(
    base_url="https://x9p8cvseqh9k7s-11434.proxy.runpod.net/v1/",
    api_key=os.getenv("OCR_API_KEY", "not-needed"),
)

# MarkItDown supplies the crop as an image to the configured vision model.
md = MarkItDown(
    llm_client=client,
    llm_model=OCR_MODEL,
    llm_prompt=ARABIC_OCR_PROMPT,
    enable_plugins=True,
)


# -------------------------------------------------------
# Arabic normalization
# -------------------------------------------------------

def normalize_arabic_text(text: str) -> str:
    """Normalize compatibility glyphs without changing RTL character order.

    Arabic presentation-form Unicode characters should be normalized, but
    reversing a Python string corrupts properly encoded Arabic OCR output.
    Rendering direction is controlled by the viewer's bidi algorithm.
    """

    return unicodedata.normalize("NFKC", text)


# -------------------------------------------------------
# LayoutParser helpers
# -------------------------------------------------------

def _load_pdf_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import optional PDF/OCR packages only when a PDF is processed."""

    try:
        import fitz  # PyMuPDF
        import layoutparser as lp
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "PDF layout OCR needs PyMuPDF, Pillow, and LayoutParser. Install "
            'them with: pip install pymupdf pillow "layoutparser[layoutmodels]"'
        ) from exc

    return fitz, lp, Image, ImageDraw


def _download_pub_laynet_weights() -> Path:
    """Download the default EfficientDet weights with a Windows-safe name."""

    LAYOUT_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    weights_path = LAYOUT_MODEL_CACHE / PUBLAYNET_WEIGHTS_NAME
    if weights_path.is_file() and weights_path.stat().st_size > 1_000_000:
        return weights_path

    partial_path = weights_path.with_suffix(weights_path.suffix + ".part")
    partial_path.unlink(missing_ok=True)
    print(f"  Downloading layout model to {weights_path}")
    try:
        with urlopen(PUBLAYNET_WEIGHTS_URL, timeout=60) as response, partial_path.open("wb") as file:
            shutil.copyfileobj(response, file)
        partial_path.replace(weights_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    return weights_path


def _build_pub_laynet_model(lp: Any, weights_path: Path) -> Any:
    """Load the published checkpoint with PyTorch 2.6+ safe-loading support.

    ``timm`` creates its own ``safe_globals`` context and therefore ignores the
    global LayoutParser/PyTorch allowlist.  Patch only that loader and only for
    this model initialization so the checkpoint's data-only metadata is
    recognized; ``weights_only=True`` remains enabled.
    """

    import numpy as np
    import timm.models._helpers as timm_helpers
    import torch

    original_torch_load = timm_helpers._torch_load
    numpy_core = np._core if hasattr(np, "_core") else np.core
    dtype_metadata = {
        type(np.dtype(dtype_name))
        for dtype_name in (
            "bool",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    }

    def load_pub_laynet_checkpoint(
        checkpoint_path: str, map_location: str = "cpu", weights_only: bool = True
    ) -> Any:
        safe_metadata = [
            Namespace,
            np.dtype,
            # The 2021 checkpoint uses NumPy's pre-2.0 module path, whereas
            # NumPy 2 exposes this callable as ``numpy._core``.
            (numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            *dtype_metadata,
        ]
        with torch.serialization.safe_globals(safe_metadata):
            return torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=weights_only,
            )

    timm_helpers._torch_load = load_pub_laynet_checkpoint
    try:
        return lp.EfficientDetLayoutModel(
            config_path="tf_efficientdet_d0",
            model_path=str(weights_path),
            label_map=PUBLAYNET_LABEL_MAP,
            extra_config={"output_confidence_threshold": LAYOUT_SCORE_THRESHOLD},
        )
    finally:
        timm_helpers._torch_load = original_torch_load


def _load_layout_model(lp: Any) -> Any:
    try:
        if LAYOUT_MODEL_URI == DEFAULT_LAYOUT_MODEL_URI:
            # Supplying a local model path bypasses LayoutParser's invalid
            # ``...?dl=1.lock`` filename on Windows.
            return _build_pub_laynet_model(lp, _download_pub_laynet_weights())

        return lp.AutoLayoutModel(LAYOUT_MODEL_URI)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load LayoutParser model {LAYOUT_MODEL_URI!r}. "
            "Ensure LayoutParser's EfficientDet dependencies and model weights "
            "are available, or set LAYOUT_MODEL_URI to another installed model."
        ) from exc


def _box_from_block(block: Any, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Return a clamped integer Pillow crop box for one LayoutParser block."""

    x1, y1, x2, y2 = block.coordinates
    left = max(0, min(image_width, int(x1)))
    top = max(0, min(image_height, int(y1)))
    right = max(left + 1, min(image_width, int(x2)))
    bottom = max(top + 1, min(image_height, int(y2)))
    return left, top, right, bottom


def _blocks_in_arabic_reading_order(
    blocks: Iterable[Any], image_height: int
) -> list[Any]:
    """Sort blocks top-to-bottom and right-to-left within the same visual row."""

    row_tolerance = max(18, int(image_height * 0.012))
    rows: list[list[Any]] = []

    for block in sorted(blocks, key=lambda item: item.coordinates[1]):
        block_top = block.coordinates[1]
        if not rows or abs(block_top - rows[-1][0].coordinates[1]) > row_tolerance:
            rows.append([block])
        else:
            rows[-1].append(block)

    return [
        block
        for row in rows
        for block in sorted(row, key=lambda item: item.coordinates[0], reverse=True)
    ]


def _draw_layout_preview(image: Any, blocks: list[Any], target: Path, image_draw: Any) -> None:
    """Save an annotated page image without requiring OpenCV."""

    preview = image.copy()
    draw = image_draw.Draw(preview)

    for index, block in enumerate(blocks, start=1):
        box = _box_from_block(block, *preview.size)
        block_type = str(getattr(block, "type", "Text"))
        draw.rectangle(box, outline="#d00000", width=4)
        draw.text((box[0] + 4, max(0, box[1] - 16)), f"{index}: {block_type}", fill="#d00000")

    preview.save(target)


def _serialize_block(block: Any, box: tuple[int, int, int, int], order: int) -> dict[str, Any]:
    score = getattr(block, "score", None)
    return {
        "reading_order": order,
        "type": str(getattr(block, "type", "Text")),
        "confidence": round(float(score), 4) if score is not None else None,
        "bbox": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
    }


# -------------------------------------------------------
# OCR PDF
# -------------------------------------------------------

def process_pdf(pdf_path: Path, output_dir: Path) -> str:
    """Run LayoutParser segmentation and MarkItDown OCR for a PDF.

    The companion JSON has the bounding boxes, labels, confidence scores, and
    region-level Markdown, making the detected PDF layout reusable downstream.
    """

    fitz, lp, Image, ImageDraw = _load_pdf_dependencies()
    layout_model = _load_layout_model(lp)
    layout_dir = output_dir / f"{pdf_path.stem}_layout"
    layout_dir.mkdir(exist_ok=True)

    layout_document: dict[str, Any] = {
        "source_file": str(pdf_path),
        "dpi": PDF_OCR_DPI,
        "layout_model": LAYOUT_MODEL_URI,
        "pages": [],
    }
    markdown_pages: list[str] = []

    with tempfile.TemporaryDirectory(prefix=f"{pdf_path.stem}_ocr_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        pdf_document = fitz.open(pdf_path)

        try:
            scale = PDF_OCR_DPI / 72
            matrix = fitz.Matrix(scale, scale)

            for page_number, page in enumerate(pdf_document, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                page_image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                page_width, page_height = page_image.size

                raw_blocks = [
                    block
                    for block in layout_model.detect(page_image)
                    if getattr(block, "score", 1.0) >= LAYOUT_SCORE_THRESHOLD
                ]
                blocks = _blocks_in_arabic_reading_order(raw_blocks, page_height)

                # Do not silently omit a page when the detector finds nothing.
                if not blocks:
                    blocks = [
                        lp.TextBlock(
                            lp.Rectangle(0, 0, page_width, page_height),
                            type="Text",
                            score=1.0,
                        )
                    ]

                preview_path = layout_dir / f"page_{page_number:04d}.png"
                _draw_layout_preview(page_image, blocks, preview_path, ImageDraw)

                page_layout: dict[str, Any] = {
                    "page_number": page_number,
                    "image_size": {"width": page_width, "height": page_height},
                    "preview": str(preview_path),
                    "blocks": [],
                }
                page_markdown: list[str] = []

                for order, block in enumerate(blocks, start=1):
                    box = _box_from_block(block, page_width, page_height)
                    crop_path = temp_dir / f"page_{page_number:04d}_block_{order:03d}.png"
                    page_image.crop(box).save(crop_path)

                    try:
                        region_markdown = md.convert(str(crop_path)).text_content.strip()
                    except Exception as exc:
                        raise RuntimeError(
                            "MarkItDown OCR failed for "
                            f"page {page_number}, block {order}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

                    block_layout = _serialize_block(block, box, order)
                    block_layout["ocr_markdown"] = region_markdown
                    page_layout["blocks"].append(block_layout)

                    if region_markdown:
                        page_markdown.append(region_markdown)

                layout_document["pages"].append(page_layout)
                markdown_pages.append(
                    f"<!-- PAGE {page_number} -->\n\n" + "\n\n".join(page_markdown)
                )
        finally:
            pdf_document.close()

    layout_path = output_dir / f"{pdf_path.stem}.layout.json"
    layout_path.write_text(
        json.dumps(layout_document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Layout JSON: {layout_path}")
    print(f"  Layout previews: {layout_dir}")

    return "\n\n".join(markdown_pages).strip()


# -------------------------------------------------------
# Office documents
# -------------------------------------------------------

def process_office_document(path: Path) -> str:
    return md.convert(str(path)).text_content


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"OCR model: {OCR_MODEL}")

    for file in INPUT_DIR.iterdir():
        if not file.is_file():
            continue

        try:
            print(f"Processing {file.name}")

            if file.suffix.lower() == ".pdf":
                text = process_pdf(file, OUTPUT_DIR)
            else:
                text = process_office_document(file)

            output_file = OUTPUT_DIR / f"{file.stem}.md"
            output_file.write_text(normalize_arabic_text(text), encoding="utf-8")

            print("✓ Done")
        except Exception as exc:
            print(f"✗ Failed: {file.name}")
            print(exc)
            if "model" in str(exc).lower() and "does not exist" in str(exc).lower():
                print(
                    "Stopping because the configured OCR model is unavailable. "
                    "Set OCR_MODEL to the exact name exposed by your RunPod deployment."
                )
                break


if __name__ == "__main__":
    main()
