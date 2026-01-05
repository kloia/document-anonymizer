"""
PDF Processing Module

Converts PDF files to images and creates PDFs from processed images.
"""

import io
import logging
from pathlib import Path
from typing import List

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Convert PDF file to high-resolution images page by page.

    Args:
        pdf_path: Path to the PDF file
        dpi: Output resolution (default 300 DPI)

    Returns:
        List of numpy arrays in OpenCV format (BGR)

    Raises:
        FileNotFoundError: If PDF file is not found
        Exception: If error occurs during PDF processing
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    images = []

    try:
        doc = fitz.open(str(pdf_path))
        logger.info(f"PDF opened: {pdf_path.name} ({len(doc)} pages)")

        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)

            img_data = np.frombuffer(pix.samples, dtype=np.uint8)

            if pix.n == 4:  # RGBA
                img = img_data.reshape(pix.height, pix.width, 4)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:  # RGB
                img = img_data.reshape(pix.height, pix.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:  # Grayscale
                img = img_data.reshape(pix.height, pix.width)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            images.append(img)
            logger.debug(f"Page {page_num + 1} rendered: {img.shape}")

        doc.close()
        logger.info(f"PDF converted to images: {len(images)} pages")

    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise

    return images


def images_to_pdf(images: List[np.ndarray], output_path: str, compression: bool = True) -> bool:
    """
    Convert list of images to PDF file.

    Args:
        images: List of numpy arrays in OpenCV format (BGR)
        output_path: Path for output PDF file
        compression: Use JPEG compression (default True)

    Returns:
        True if successful

    Raises:
        ValueError: If image list is empty
        Exception: If error occurs during PDF creation
    """
    if not images:
        raise ValueError("Image list is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open()

        for idx, img in enumerate(images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            img_bytes = io.BytesIO()

            if compression:
                pil_img.save(img_bytes, format="JPEG", quality=95)
            else:
                pil_img.save(img_bytes, format="PNG")

            img_bytes.seek(0)

            img_rect = fitz.Rect(0, 0, pil_img.width, pil_img.height)
            page = doc.new_page(width=pil_img.width, height=pil_img.height)
            page.insert_image(img_rect, stream=img_bytes.getvalue())

            logger.debug(f"Page {idx + 1} added to PDF")

        doc.save(str(output_path))
        doc.close()

        logger.info(f"PDF created: {output_path}")
        return True

    except Exception as e:
        logger.error(f"PDF creation error: {e}")
        raise


def get_pdf_info(pdf_path: str) -> dict:
    """
    Return information about PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing PDF information
    """
    pdf_path = Path(pdf_path)

    try:
        doc = fitz.open(str(pdf_path))

        info = {
            "path": str(pdf_path),
            "filename": pdf_path.name,
            "page_count": len(doc),
            "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
            "metadata": doc.metadata,
        }

        if len(doc) > 0:
            first_page = doc[0]
            rect = first_page.rect
            info["page_width"] = rect.width
            info["page_height"] = rect.height

        doc.close()
        return info

    except Exception as e:
        logger.error(f"Failed to get PDF info: {e}")
        raise
