import json

import shutil

import subprocess

import tempfile
import time

from dataclasses import dataclass, asdict

from pathlib import Path

from typing import List, Optional

import fitz  # pip install pymupdf

from ollama import Client  # pip install ollama

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

SUPPORTED_DOC_EXTS = {".pdf", ".docx", *SUPPORTED_IMAGE_EXTS}


@dataclass
class OCRPageResult:
    source_file: str

    page_number: int

    image_path: str

    markdown: str


class DeepSeekOCRProcessor:

    def __init__(

            self,

            client: Client,

            model: str = "deepseek-ocr:3b",

            prompt: str = "<|grounding|>Convert the document to markdown.",

            num_predict: int = 4096,

            dpi: int = 220,

            temp_dir: Optional[str] = None,

    ):

        self.client = client

        self.model = model

        self.prompt = prompt

        self.num_predict = num_predict

        self.dpi = dpi

        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="deepseek_ocr_"))

    def ensure_dependencies(self):

        if shutil.which("soffice") is None and shutil.which("libreoffice") is None:
            print("Warning: LibreOffice is not installed. DOCX files will not work.")

        if shutil.which("ollama") is None:
            print("Warning: Ollama CLI not found in PATH. Make sure Ollama server is running.")

    def process(self, input_path: str, output_dir: str) -> List[OCRPageResult]:

        input_file = Path(input_path).expanduser().resolve()

        output_path = Path(output_dir).expanduser().resolve()

        output_path.mkdir(parents=True, exist_ok=True)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        ext = input_file.suffix.lower()

        if ext not in SUPPORTED_DOC_EXTS:
            raise ValueError(f"Unsupported file type: {ext}")

        page_images = self._prepare_images(input_file)

        results: List[OCRPageResult] = []

        for idx, image_path in enumerate(page_images, start=1):
            markdown = self._ocr_image(image_path)

            markdown = self._postprocess_markdown(markdown)

            results.append(

                OCRPageResult(

                    source_file=str(input_file),

                    page_number=idx,

                    image_path=str(image_path),

                    markdown=markdown,

                )

            )

        self._write_outputs(results, output_path, input_file.stem)

        return results

    def _prepare_images(self, input_file: Path) -> List[Path]:

        ext = input_file.suffix.lower()

        if ext in SUPPORTED_IMAGE_EXTS:
            return [input_file]

        if ext == ".pdf":
            return self._pdf_to_images(input_file)

        if ext == ".docx":
            pdf_path = self._docx_to_pdf(input_file)

            return self._pdf_to_images(pdf_path)

        raise ValueError(f"Unsupported file type: {ext}")

    def _docx_to_pdf(self, docx_path: Path) -> Path:

        out_dir = self.temp_dir / "docx_pdf"

        out_dir.mkdir(parents=True, exist_ok=True)

        soffice_cmd = shutil.which("soffice") or shutil.which("libreoffice")

        if not soffice_cmd:
            raise RuntimeError(

                "LibreOffice is required for DOCX support.\n"

                "Ubuntu/Debian: sudo apt-get install -y libreoffice"

            )

        cmd = [

            soffice_cmd,

            "--headless",

            "--convert-to",

            "pdf",

            "--outdir",

            str(out_dir),

            str(docx_path),

        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(

                f"DOCX to PDF conversion failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

            )

        pdf_path = out_dir / f"{docx_path.stem}.pdf"

        if not pdf_path.exists():
            raise RuntimeError(f"Converted PDF not found: {pdf_path}")

        return pdf_path

    def _pdf_to_images(self, pdf_path: Path) -> List[Path]:

        images_dir = self.temp_dir / f"{pdf_path.stem}_pages"

        images_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)

        image_paths = []

        zoom = self.dpi / 72.0

        matrix = fitz.Matrix(zoom, zoom)

        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            img_path = images_dir / f"page_{i + 1:04d}.png"

            pix.save(str(img_path))

            image_paths.append(img_path)

        doc.close()

        return image_paths

    def _ocr_image(self, image_path: Path) -> str:

        response = self.client.chat(

            model=self.model,

            messages=[

                {

                    "role": "user",

                    "content": self.prompt,

                    "images": [str(image_path)],

                }

            ],

            options={

                "num_predict": self.num_predict,

            },

        )

        return response["message"]["content"]

    def _postprocess_markdown(self, text: str) -> str:

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        lines = [line.rstrip() for line in text.splitlines()]

        cleaned = "\n".join(lines).strip()

        deduped_lines = []

        last = None

        repeat_count = 0

        for line in cleaned.splitlines():

            if line == last:

                repeat_count += 1

            else:

                repeat_count = 0

            if repeat_count < 2:
                deduped_lines.append(line)

            last = line

        return "\n".join(deduped_lines).strip()

    def _write_outputs(self, results: List[OCRPageResult], output_dir: Path, base_name: str):

        md_path = output_dir / f"{base_name}.md"

        json_path = output_dir / f"{base_name}.json"

        with open(md_path, "w", encoding="utf-8") as f:

            for r in results:

                if len(results) > 1:
                    f.write(f"\n\n<!-- PAGE {r.page_number} -->\n\n")

                f.write(r.markdown)

                f.write("\n")

        with open(json_path, "w", encoding="utf-8") as f:

            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


def main():
    ollama_url = "https://r4pl207radupn7-11434.proxy.runpod.net/"

    INPUT_DIR = Path("input")

    output_dir = r"./output"

    model = "deepseek-ocr"

    prompt = """<|grounding|>Convert the document to markdown.

    Additional OCR constraints:
    1. Primary language is Arabic. Preserve RTL alignment, diacritics, and connected letter forms exactly.
    2. Preserve all layout elements: paragraphs, indentation, bullet points, and table structures.
    3. Transcribe text verbatim. No summarization.
    4. This document contains zero Chinese characters. Discard any CJK hallucinations.
    5. Output ONLY raw markdown text."""

    num_predict = 4096

    dpi = 220

    client = Client(host=ollama_url)

    processor = DeepSeekOCRProcessor(

        client=client,

        model=model,

        prompt=prompt,

        num_predict=num_predict,

        dpi=dpi,

    )

    processor.ensure_dependencies()
    for file in INPUT_DIR.iterdir():
        total_start = time.perf_counter()
        results = processor.process(str(file), output_dir)
        total_end = time.perf_counter()

        print(f"⏱️ Total Script Execution Time: {total_end - total_start:.2f} seconds")

        print(f"Processed {len(results)} page(s).")

        print(f"Markdown saved to: {Path(output_dir).resolve() / (Path(file).stem + '.md')}")

        print(f"JSON saved to: {Path(output_dir).resolve() / (Path(file).stem + '.json')}")


if __name__ == "__main__":
    main()


