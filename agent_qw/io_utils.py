"""File I/O utilities: PDF to image conversion, path helpers, file reading."""

import logging
from pathlib import Path
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """Convert a PDF file to a list of PIL Images (one per page).

    Uses PyMuPDF (fitz) for rendering — no poppler required.
    Falls back to pdf2image if fitz is unavailable.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. 300 is good for OCR quality.

    Returns:
        List of PIL Images in RGB mode.
    """
    # Primary: PyMuPDF (no system dependencies needed)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        # Calculate zoom factor: fitz uses matrix scaling, default is 72 DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        logger.debug("Rendered %d pages via PyMuPDF at %d DPI", len(images), dpi)
        return images

    except ImportError:
        logger.debug("PyMuPDF not available; trying pdf2image (requires poppler)")

    # Fallback: pdf2image (requires poppler)
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        return [img.convert("RGB") for img in images]
    except ImportError:
        raise ImportError(
            "Neither PyMuPDF nor pdf2image is installed. "
            "Install with: pip install pymupdf"
        )


def convert_path_to_images(input_path: str) -> List[Image.Image]:
    """Convert an input path (PDF or image file) to a list of PIL Images.

    Args:
        input_path: Path to a PDF file or an image file (png, jpg, tiff, etc.).

    Returns:
        List of PIL Images.
    """
    ext = Path(input_path).suffix.lower()

    if ext == ".pdf":
        return convert_pdf_to_images(input_path)
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
        img = Image.open(input_path).convert("RGB")
        return [img]
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def collect_input_files(input_path: str) -> List[str]:
    """Collect all supported input files from a path.

    If input_path is a file, returns [file].
    If input_path is a directory, returns all PDF/image files found.

    Args:
        input_path: Path to a file or directory.

    Returns:
        Sorted list of absolute file paths.
    """
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() in supported:
            return [str(path.resolve())]
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    if path.is_dir():
        files = []
        for ext in supported:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(str(f.resolve()) for f in files)

    raise FileNotFoundError(f"Input path not found: {input_path}")
