"""
Image Masking Module

Masks sensitive areas in images using various methods.
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import validate_bbox

logger = logging.getLogger(__name__)


class ImageMasker:
    """
    Image masking class for signatures and stamps.

    Uses contour-based masking which is non-reversible and GDPR/KVKK compliant.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the masker."""
        self.config = config or {}

    def mask_signature_stamp_contour(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Mask signature/stamp areas using contour detection.

        Uses non-reversible black fill on detected strokes for GDPR/KVKK compliance.
        """
        x1, y1, x2, y2 = self._validate_bbox(bbox, image.shape)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return image

        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding for ink detection
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Block size
            3,  # C constant
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours found, fill entire bbox with opaque color
        if not contours:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
            logger.debug(f"Contour mask (no contours, full fill): ({x1}, {y1}) - ({x2}, {y2})")
            return image

        # Create mask from contours
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, -1)

        # Dilate mask slightly to ensure complete coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Apply black fill where mask is active
        image[y1:y2, x1:x2][mask > 0] = (0, 0, 0)

        logger.debug(f"Contour mask ({len(contours)} contours): ({x1}, {y1}) - ({x2}, {y2})")
        return image

    def _validate_bbox(
        self, bbox: Tuple[int, int, int, int], shape: tuple
    ) -> Tuple[int, int, int, int]:
        """Validate bbox coordinates."""
        return validate_bbox(bbox, shape)
