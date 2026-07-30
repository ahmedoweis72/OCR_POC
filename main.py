from pathlib import Path
import unicodedata
from markitdown import MarkItDown
from openai import OpenAI
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from markitdown import MarkItDown

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

# Change if Tesseract is not in PATH


client = OpenAI(
    api_key="sk-52b7ce7d2bad4a4ca47942b170d6dcfd",
    base_url="https://api.deepseek.com"
)
md = MarkItDown(
    llm_client=client,
    llm_model="deepseek-v4-pro"
)
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

LINE_THRESHOLD = 12


# -------------------------------------------------------
# Arabic normalization
# -------------------------------------------------------

def normalize_arabic_text(text: str) -> str:
    lines = []

    for line in text.splitlines():
        if any("\uFB50" <= ch <= "\uFEFF" for ch in line):
            line = unicodedata.normalize("NFKC", line)
            line = line[::-1]

        lines.append(line)

    return "\n".join(lines)


# -------------------------------------------------------
# OCR PDF
# -------------------------------------------------------

def process_pdf(pdf_path: Path) -> str:

    result = md.convert(str(pdf_path))

    return result.text_content;


# -------------------------------------------------------
# Office documents
# -------------------------------------------------------

def process_office_document(path: Path) -> str:
    result = md.convert(str(path))

    return result.text_content


# -------------------------------------------------------
# Main
# -------------------------------------------------------

for file in INPUT_DIR.iterdir():

    if not file.is_file():
        continue

    try:

        print(f"Processing {file.name}")

        if file.suffix.lower() == ".pdf":

            text = process_pdf(file)

        else:

            text = process_office_document(file)

        text = normalize_arabic_text(text)

        output_file = OUTPUT_DIR / f"{file.stem}.md"

        output_file.write_text(
            text,
            encoding="utf-8"
        )

        print("✓ Done")

    except Exception as ex:

        print(f"✗ Failed: {file.name}")
        print(ex)
