"""Tests for pdf_handler module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from document_anonymizer.pdf_handler import (
    get_pdf_info,
    images_to_pdf,
    pdf_to_images,
)


class TestPdfToImages:
    """Tests for pdf_to_images function."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            pdf_to_images("/nonexistent/path/file.pdf")

    def test_invalid_path_type(self):
        """Should handle Path objects."""
        with pytest.raises(FileNotFoundError):
            pdf_to_images(Path("/nonexistent/path/file.pdf"))


class TestImagesToPdf:
    """Tests for images_to_pdf function."""

    def test_empty_images_list(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="Image list is empty"):
            images_to_pdf([], "/tmp/test.pdf")

    def test_creates_output_directory(self):
        """Should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "output.pdf"

            # Create a simple test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

            result = images_to_pdf([test_image], str(output_path))

            assert result is True
            assert output_path.exists()

    def test_single_image(self):
        """Should create PDF from single image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "single.pdf"
            test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128

            result = images_to_pdf([test_image], str(output_path))

            assert result is True
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_multiple_images(self):
        """Should create PDF from multiple images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "multi.pdf"
            images = [
                np.ones((100, 100, 3), dtype=np.uint8) * 255,
                np.ones((100, 100, 3), dtype=np.uint8) * 128,
                np.ones((100, 100, 3), dtype=np.uint8) * 64,
            ]

            result = images_to_pdf(images, str(output_path))

            assert result is True
            assert output_path.exists()

    def test_compression_option(self):
        """Should respect compression option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compressed_path = Path(tmpdir) / "compressed.pdf"
            uncompressed_path = Path(tmpdir) / "uncompressed.pdf"
            test_image = np.ones((500, 500, 3), dtype=np.uint8) * 200

            images_to_pdf([test_image], str(compressed_path), compression=True)
            images_to_pdf([test_image], str(uncompressed_path), compression=False)

            # Uncompressed (PNG) should typically be larger
            assert compressed_path.exists()
            assert uncompressed_path.exists()

    def test_different_image_sizes(self):
        """Should handle different image sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "varied.pdf"
            images = [
                np.ones((100, 200, 3), dtype=np.uint8) * 255,  # Landscape
                np.ones((300, 150, 3), dtype=np.uint8) * 128,  # Portrait
                np.ones((200, 200, 3), dtype=np.uint8) * 64,  # Square
            ]

            result = images_to_pdf(images, str(output_path))

            assert result is True


class TestGetPdfInfo:
    """Tests for get_pdf_info function."""

    def test_file_not_found(self):
        """Should raise exception for missing file."""
        with pytest.raises(Exception):
            get_pdf_info("/nonexistent/path/file.pdf")

    def test_info_from_created_pdf(self):
        """Should return correct info for created PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_info.pdf"
            test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255

            images_to_pdf([test_image], str(pdf_path))

            info = get_pdf_info(str(pdf_path))

            assert info["filename"] == "test_info.pdf"
            assert info["page_count"] == 1
            assert "file_size_mb" in info
            assert "page_width" in info
            assert "page_height" in info

    def test_multi_page_pdf_info(self):
        """Should report correct page count for multi-page PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "multi_page.pdf"
            images = [np.ones((100, 100, 3), dtype=np.uint8) for _ in range(5)]

            images_to_pdf(images, str(pdf_path))

            info = get_pdf_info(str(pdf_path))

            assert info["page_count"] == 5


class TestRoundTrip:
    """Tests for PDF round-trip (create -> read -> verify)."""

    def test_basic_round_trip(self):
        """Created PDF should be readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "round_trip.pdf"

            # Create test images with different content
            original_images = [
                np.zeros((200, 200, 3), dtype=np.uint8),  # Black
                np.ones((200, 200, 3), dtype=np.uint8) * 255,  # White
            ]
            original_images[0][50:150, 50:150] = [255, 0, 0]  # Red square

            # Create PDF
            images_to_pdf(original_images, str(pdf_path))

            # Read back
            read_images = pdf_to_images(str(pdf_path))

            assert len(read_images) == 2
            assert all(img.shape[2] == 3 for img in read_images)  # All BGR

    def test_high_dpi_round_trip(self):
        """High DPI conversion should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "high_dpi.pdf"
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128

            images_to_pdf([test_image], str(pdf_path))

            # Read at high DPI
            read_images = pdf_to_images(str(pdf_path), dpi=600)

            assert len(read_images) == 1
            # Higher DPI should produce larger image
            assert read_images[0].shape[0] > 100
            assert read_images[0].shape[1] > 100

    def test_low_dpi_round_trip(self):
        """Low DPI conversion should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "low_dpi.pdf"
            test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128

            images_to_pdf([test_image], str(pdf_path))

            # Read at low DPI
            read_images = pdf_to_images(str(pdf_path), dpi=72)

            assert len(read_images) == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_grayscale_conversion(self):
        """Should handle grayscale images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "grayscale.pdf"

            # Create a grayscale-like BGR image
            gray_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 128

            result = images_to_pdf([gray_bgr], str(pdf_path))

            assert result is True
            read_images = pdf_to_images(str(pdf_path))
            assert len(read_images) == 1

    def test_large_image(self):
        """Should handle large images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "large.pdf"

            # Large image (A4 at 300 DPI approximately)
            large_image = np.ones((3508, 2480, 3), dtype=np.uint8) * 200

            result = images_to_pdf([large_image], str(pdf_path))

            assert result is True
            assert pdf_path.exists()

    def test_small_image(self):
        """Should handle very small images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "small.pdf"
            small_image = np.ones((10, 10, 3), dtype=np.uint8) * 128

            result = images_to_pdf([small_image], str(pdf_path))

            assert result is True
