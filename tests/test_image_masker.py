"""Tests for image_masker module."""

import numpy as np
import pytest

from document_anonymizer.image_masker import ImageMasker


class TestImageMaskerInit:
    """Tests for ImageMasker initialization."""

    def test_default_initialization(self):
        """Should initialize with default config."""
        masker = ImageMasker()
        assert masker.config == {}

    def test_custom_config(self):
        """Should accept custom config."""
        config = {"test": "value"}
        masker = ImageMasker(config=config)
        assert masker.config == config


class TestContourMasking:
    """Tests for contour-based masking."""

    @pytest.fixture
    def masker(self):
        return ImageMasker()

    @pytest.fixture
    def sample_image(self):
        """Create a sample BGR image with some content."""
        # White background
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        # Add some black lines (simulate signature)
        img[50:60, 50:150] = [0, 0, 0]
        img[70:80, 60:140] = [0, 0, 0]
        img[90:100, 70:130] = [0, 0, 0]
        return img

    def test_mask_signature_stamp_contour_basic(self, masker, sample_image):
        """Should mask area with contours."""
        bbox = (40, 40, 160, 110)
        result = masker.mask_signature_stamp_contour(sample_image.copy(), bbox)

        # Result should be same shape
        assert result.shape == sample_image.shape

        # Masked area should have black pixels where contours were
        crop = result[40:110, 40:160]
        assert np.any(crop == 0)

    def test_mask_empty_bbox(self, masker, sample_image):
        """Should handle empty bbox gracefully."""
        bbox = (0, 0, 0, 0)
        result = masker.mask_signature_stamp_contour(sample_image.copy(), bbox)
        # Should return original image unchanged
        assert result.shape == sample_image.shape

    def test_mask_out_of_bounds_bbox(self, masker, sample_image):
        """Should handle out of bounds bbox."""
        h, w = sample_image.shape[:2]
        bbox = (w - 10, h - 10, w + 100, h + 100)
        result = masker.mask_signature_stamp_contour(sample_image.copy(), bbox)
        assert result.shape == sample_image.shape

    def test_mask_no_contours(self, masker):
        """Should fill entire bbox when no contours found."""
        # Pure white image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bbox = (10, 10, 50, 50)
        result = masker.mask_signature_stamp_contour(img.copy(), bbox)

        # Masked area should be black
        crop = result[10:50, 10:50]
        assert np.all(crop == 0)


class TestBboxValidation:
    """Tests for bbox validation helper."""

    @pytest.fixture
    def masker(self):
        return ImageMasker()

    def test_validate_bbox_normal(self, masker):
        """Should return valid bbox unchanged."""
        shape = (100, 200, 3)
        bbox = (10, 20, 50, 60)
        result = masker._validate_bbox(bbox, shape)
        assert result == (10, 20, 50, 60)

    def test_validate_bbox_clamps_to_bounds(self, masker):
        """Should clamp bbox to image bounds."""
        shape = (100, 200, 3)
        bbox = (-10, -20, 250, 150)
        x1, y1, x2, y2 = masker._validate_bbox(bbox, shape)

        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 200
        assert y2 <= 100

    def test_validate_bbox_swapped_coords(self, masker):
        """Should handle swapped coordinates."""
        shape = (100, 200, 3)
        bbox = (50, 60, 10, 20)  # Swapped
        x1, y1, x2, y2 = masker._validate_bbox(bbox, shape)

        # Should ensure x2 > x1 and y2 > y1
        assert x2 > x1
        assert y2 > y1
