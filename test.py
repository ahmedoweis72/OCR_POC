import requests
import base64
import os
import json
from typing import Optional, Union
from pathlib import Path


def document_ocr(
        query: Union[str, Path, bytes],
        *,
        model: str = "deepseek-v4-pro",  # Keep your model
        base_url: str = "https://5o0iqnpt4j1zst-11434.proxy.runpod.net/",  # Your RunPod URL
        timeout_s: float = 120.0,
) -> Optional[str]:
    """
    Perform OCR on PDF/document using DeepSeek on RunPod.

    Args:
        query: Text query, PDF file path, or PDF bytes
        model: DeepSeek model name
        base_url: Your RunPod endpoint
        timeout_s: Timeout in seconds

    Returns:
        Extracted text or None on failure
    """

    # Convert input to base64 if it's a file
    content = None
    is_document = False

    if isinstance(query, (str, Path)):
        path = Path(query)
        if path.exists() and path.is_file():
            # Check if it's a PDF or image
            if path.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                with open(path, "rb") as f:
                    file_bytes = f.read()
                    content = base64.b64encode(file_bytes).decode("utf-8")
                    is_document = True
            else:
                # Treat as text query
                content = query.strip()
        else:
            # Treat as text query
            content = str(query).strip()
    elif isinstance(query, bytes):
        # Treat as document bytes
        content = base64.b64encode(query).decode("utf-8")
        is_document = True
    else:
        content = str(query).strip()

    # Prepare the request for RunPod
    url = f"{base_url.rstrip('/')}/api/chat"

    # Build the prompt for OCR
    if is_document:
        # For document OCR, use the image/document in the prompt
        system_prompt = (
            "You are an OCR assistant. Extract ALL text from the provided document. "
            "Preserve formatting, paragraphs, and structure. "
            "Output ONLY the extracted text, no explanations or labels."
        )
        user_content = f"Extract all text from this document. Document data: {content[:100]}..."  # Truncate for display

        # For RunPod with vision models, you might need to use the image URL format
        # Since this is a chat endpoint, we'll use the base64 image format
        user_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:application/pdf;base64,{content}"
            }
        }
        # For text models without vision, send as text with document marker
        # user_content = f"[DOCUMENT_START]\n{content}\n[DOCUMENT_END]\nExtract all text from this document."
    else:
        # Text query - translation as before
        system_prompt = (
            "Translate the user text to English for document search. "
            "Output ONLY the English translation on one line — no labels or quotes."
        )
        user_content = content

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content
            },
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 4096},  # Increased for longer OCR output
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()

        response_data = resp.json()
        text = (response_data.get("message") or {}).get("content", "").strip()

        # For OCR, we want all text, not just first line
        if is_document:
            # Return the full extracted text
            return text if text else None

        # For translation, return first non-empty line
        for line in text.splitlines():
            line = line.strip().strip('"').strip("'")
            if line:
                return line

        return None

    except requests.RequestException as exc:
        print(f"⚠️ RunPod request failed: {exc}")
        if hasattr(exc, 'response') and exc.response:
            try:
                error_detail = exc.response.json()
                print(f"Error detail: {error_detail}")
            except:
                print(f"Response: {exc.response.text[:200]}")
        return None
    except Exception as exc:
        print(f"⚠️ Unexpected error: {exc}")
        return None


# Specialized function for PDF OCR with DeepSeek
def pdf_ocr(
        pdf_path: Union[str, Path],
        *,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://5o0iqnpt4j1zst-11434.proxy.runpod.net/",
        timeout_s: float = 180.0,
        chunk_size: int = 10,  # Pages per chunk for large PDFs
) -> Optional[str]:
    """
    Specialized PDF OCR function with chunking support.
    """
    path = Path(pdf_path)
    if not path.exists():
        print(f"⚠️ File not found: {pdf_path}")
        return None

    # For large PDFs, you might need to process in chunks
    # This is a simplified version - you may need to use PyPDF2 or similar
    # to split PDFs into pages

    try:
        import PyPDF2
        from io import BytesIO

        with open(path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            total_pages = len(pdf_reader.pages)

            if total_pages <= chunk_size:
                # Small PDF, process whole file
                return document_ocr(path, model=model, base_url=base_url, timeout_s=timeout_s)
            else:
                # Large PDF, process in chunks
                all_text = []
                for start_page in range(0, total_pages, chunk_size):
                    end_page = min(start_page + chunk_size, total_pages)
                    print(f"Processing pages {start_page + 1}-{end_page} of {total_pages}")

                    # Create a PDF with just these pages
                    pdf_writer = PyPDF2.PdfWriter()
                    for page_num in range(start_page, end_page):
                        pdf_writer.add_page(pdf_reader.pages[page_num])

                    pdf_bytes = BytesIO()
                    pdf_writer.write(pdf_bytes)

                    # Process this chunk
                    chunk_text = document_ocr(
                        pdf_bytes.getvalue(),
                        model=model,
                        base_url=base_url,
                        timeout_s=timeout_s
                    )

                    if chunk_text:
                        all_text.append(chunk_text)

                return "\n\n".join(all_text) if all_text else None

    except ImportError:
        print("⚠️ PyPDF2 not installed. Install with: pip install PyPDF2")
        # Fallback to processing whole file
        return document_ocr(path, model=model, base_url=base_url, timeout_s=timeout_s)
    except Exception as exc:
        print(f"⚠️ PDF processing error: {exc}")
        return None


# Example usage
if __name__ == "__main__":
    # For translation (original use case)
    translation = document_ocr(
        "مرحبا كيف حالك",
        base_url="https://5o0iqnpt4j1zst-11434.proxy.runpod.net/"
    )
    print(f"Translation: {translation}")

    # For PDF OCR
    ocr_text = pdf_ocr(
        "document.pdf",
        base_url="https://5o0iqnpt4j1zst-11434.proxy.runpod.net/"
    )
    print(f"OCR Result: {ocr_text}")