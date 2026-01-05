"""
Shared Utility Functions

Common utilities used across multiple modules.
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def validate_bbox(bbox: Tuple[int, int, int, int], shape: tuple) -> Tuple[int, int, int, int]:
    """
    Validate and constrain bbox to image bounds.

    Args:
        bbox: (x1, y1, x2, y2) coordinates
        shape: Image shape (height, width, ...)

    Returns:
        Validated bbox tuple
    """
    x1, y1, x2, y2 = bbox
    h, w = shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))

    return x1, y1, x2, y2


def is_signature_or_stamp(field: Dict) -> bool:
    """
    Determine if a field is a signature or stamp that requires blur masking.

    Args:
        field: Field dictionary with field_type and detection_method

    Returns:
        True if field is a signature/stamp requiring blur masking
    """
    field_type = field.get("field_type", "").lower()
    detection_method = field.get("detection_method", "")

    # Visual detection method
    is_visual_signature = detection_method == "visual_detection"

    # Explicit signature/stamp field types
    signature_stamp_types = {
        "stamp",
        "signature",
        "seal",
        "handwriting",
        "kase",
        "imza",
        "muhur",  # Turkish terms
    }
    is_signature_type = field_type in signature_stamp_types

    # LLM-detected signature/stamp (but not signatory_name, etc.)
    is_llm_signature = (
        ("signature" in field_type or "stamp" in field_type)
        and "name" not in field_type
        and "number" not in field_type
    )

    return is_visual_signature or is_signature_type or is_llm_signature
