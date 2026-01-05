"""
Sensitive Field Detection Module

LLM-first detection with pattern matching fallback.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import REFERENCE_PATTERNS, has_legal_suffix

logger = logging.getLogger(__name__)

# Confidence thresholds
CONFIDENCE_AUTO_MASK = 0.85  # Auto-mask threshold
CONFIDENCE_REVIEW = 0.60  # Manual review threshold
CONFIDENCE_IGNORE = 0.60  # Ignore below this

# Regulation mapping for pattern-detected fields
PATTERN_REGULATIONS: Dict[str, List[str]] = {
    # Global patterns
    "email": ["GDPR", "KVKK", "CCPA", "LGPD"],
    "ip_address": ["GDPR", "KVKK", "CCPA"],
    "date_iso": ["GDPR", "KVKK"],
    "date_us": ["GDPR", "KVKK"],
    "date_eu": ["GDPR", "KVKK"],
    # National ID patterns
    "ssn_us": ["CCPA", "HIPAA"],
    "nino_uk": ["GDPR"],
    "insee_fr": ["GDPR"],
    "cf_it": ["GDPR"],
    "dni_es": ["GDPR"],
    # License plate patterns
    "plate_tr": ["KVKK", "GDPR"],
    "plate_uk": ["GDPR"],
    "plate_fr": ["GDPR"],
    "plate_it": ["GDPR"],
    "plate_es": ["GDPR"],
    "plate_ru": ["GDPR"],
    # Postal code patterns (only UK - specific format)
    "postal_uk": ["GDPR"],
    # Phone patterns (country code required)
    "phone_us": ["CCPA"],
    "phone_tr": ["KVKK", "GDPR"],
    "phone_de": ["GDPR"],
    "phone_fr": ["GDPR"],
    "phone_uk": ["GDPR"],
    "phone_ru": ["GDPR"],
}

# Risk level mapping for pattern types
PATTERN_RISK_LEVELS: Dict[str, str] = {
    # Global patterns
    "email": "MEDIUM",
    "ip_address": "MEDIUM",
    "date_iso": "LOW",
    "date_us": "LOW",
    "date_eu": "LOW",
    # National IDs - CRITICAL (uniquely identifies individuals)
    "ssn_us": "CRITICAL",
    "nino_uk": "CRITICAL",
    "insee_fr": "CRITICAL",
    "cf_it": "CRITICAL",
    "dni_es": "CRITICAL",
    # License plates - HIGH (can identify vehicle owner)
    "plate_tr": "HIGH",
    "plate_uk": "HIGH",
    "plate_fr": "HIGH",
    "plate_it": "HIGH",
    "plate_es": "HIGH",
    "plate_ru": "HIGH",
    # Postal codes - LOW (only UK)
    "postal_uk": "LOW",
    # Phone patterns - MEDIUM
    "phone_us": "MEDIUM",
    "phone_tr": "MEDIUM",
    "phone_de": "MEDIUM",
    "phone_fr": "MEDIUM",
    "phone_uk": "MEDIUM",
    "phone_ru": "MEDIUM",
}


class FieldDetector:
    """
    LLM-first sensitive field detector.

    Detection strategy:
    1. If LLM API available: Use LLM for primary detection
    2. If LLM not available: Fall back to pattern matching
    3. Separate visual detection for signatures/stamps
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize field detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        detection_config = self.config.get("detection_rules", {})
        self.use_llm_classification = detection_config.get("use_llm_classification", True)
        self.use_fallback_detection = detection_config.get("use_fallback_detection", True)
        self.min_confidence = detection_config.get("min_confidence", CONFIDENCE_REVIEW)
        self.auto_mask_confidence = detection_config.get(
            "auto_mask_confidence", CONFIDENCE_AUTO_MASK
        )

        # Compile patterns for fallback
        self._patterns = {}
        for name, pattern in REFERENCE_PATTERNS.items():
            try:
                self._patterns[name] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Invalid pattern {name}: {e}")

        # LLM classifier (lazy loaded)
        self._classifier = None

        logger.debug(f"FieldDetector initialized (llm_enabled={self.use_llm_classification})")

    async def _get_detector(self):
        """Get LLM detector (lazy loading)."""
        if self._classifier is None:
            from .llm_classifier import LLMDetector

            self._classifier = LLMDetector(self.config)
        return self._classifier

    async def detect_document_language(self, image: np.ndarray) -> Dict:
        """Detect document language using LLM. Returns {'languages': [...], 'locale': '...'}."""
        default_result = {"languages": ["en"], "locale": "en_US"}

        if not self.use_llm_classification:
            logger.debug("LLM disabled, using defaults")
            return default_result

        try:
            classifier = await self._get_detector()
            result = await classifier.detect_document_language(image)
            return result
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, using defaults")
            return default_result

    async def detect_sensitive_fields(
        self, image: np.ndarray, ocr_results: List[Dict], page_num: int = 1
    ) -> Tuple[List[Dict], List[Dict]]:
        """Detect sensitive fields. Returns (auto_mask_fields, review_fields)."""
        all_detected = []

        # LLM-based unified detection
        if self.use_llm_classification:
            try:
                classifier = await self._get_detector()

                # Single unified call for text + visual detection
                unified_result = await classifier.detect_all(image, ocr_results)

                # Process text detections (with label preservation)
                for td in unified_result.get("text_detections", []):
                    field = {
                        "block_id": td.get("block_id"),
                        "text": td.get("sensitive_value", ""),  # Only the value to mask
                        "original_text": td.get("sensitive_value", ""),
                        "full_text": td.get("full_text", ""),  # Complete OCR text
                        "label": td.get("label"),  # Label to preserve (can be None)
                        "bbox": td.get("bbox"),
                        "field_type": td.get("field_type", "unknown"),
                        "confidence": td.get("confidence", 0.5),
                        "page": page_num,
                        "detection_method": td.get("detection_method", "llm_unified"),
                        "reasoning": td.get("reasoning", ""),
                        "risk_level": td.get("risk_level", "MEDIUM"),
                        "font_properties": td.get("font_properties", {}),
                    }
                    all_detected.append(field)

                # Process visual detections (signatures, stamps)
                for vd in unified_result.get("visual_detections", []):
                    field = {
                        "block_id": vd.get("element_id"),
                        "text": f"[{vd.get('field_type', 'VISUAL').upper()}]",
                        "original_text": "",
                        "bbox": vd.get("bbox"),
                        "field_type": vd.get("field_type", "signature"),
                        "confidence": vd.get("confidence", 0.8),
                        "page": page_num,
                        "detection_method": vd.get("detection_method", "llm_unified_visual"),
                        "description": vd.get("description", ""),
                        "is_visual": True,
                    }
                    all_detected.append(field)

                text_count = len(unified_result.get("text_detections", []))
                visual_count = len(unified_result.get("visual_detections", []))
                logger.info(
                    f"Page {page_num}: Unified detection - {text_count} text, {visual_count} visual"
                )

                # ALWAYS run pattern matching as supplement to catch anything LLM missed
                if self.use_fallback_detection:
                    pattern_fields = self._detect_by_patterns(ocr_results, page_num)
                    if pattern_fields:
                        logger.info(
                            f"Page {page_num}: Pattern supplement found {len(pattern_fields)} additional fields"
                        )
                        all_detected.extend(pattern_fields)

            except Exception as e:
                logger.warning(f"Unified detection failed, using fallback: {e}")
                if self.use_fallback_detection:
                    fallback_fields = self._detect_by_patterns(ocr_results, page_num)
                    all_detected.extend(fallback_fields)

        # Pattern matching only (no LLM)
        elif self.use_fallback_detection:
            fallback_fields = self._detect_by_patterns(ocr_results, page_num)
            all_detected.extend(fallback_fields)
            logger.info(f"Page {page_num}: Pattern matching detected {len(fallback_fields)} fields")

        unique_fields = self._deduplicate_fields(all_detected)

        # Separate by confidence
        auto_mask = []
        needs_review = []

        for field in unique_fields:
            confidence = field.get("confidence", 0)

            if confidence >= self.auto_mask_confidence:
                field["review_status"] = "auto_mask"
                auto_mask.append(field)
            elif confidence >= self.min_confidence:
                field["review_status"] = "needs_review"
                needs_review.append(field)
            # Ignore below min_confidence

        logger.info(
            f"Page {page_num}: {len(auto_mask)} auto-mask, {len(needs_review)} needs review"
        )

        return auto_mask, needs_review

    def _detect_by_patterns(self, ocr_results: List[Dict], page_num: int) -> List[Dict]:
        """Fallback: Detect sensitive fields using regex patterns."""
        detected = []

        for block in ocr_results:
            text = block.get("text", "")
            bbox = block.get("bbox")
            block_id = block.get("block_id")
            confidence = block.get("confidence", 0.5)

            if not text or len(text) < 3:
                continue

            for pattern_name, pattern in self._patterns.items():
                match = pattern.search(text)
                if match:
                    # Additional validation for common false positives
                    if self._is_likely_false_positive(text, pattern_name):
                        continue

                    # Extract the matched text, not the entire block
                    matched_text = match.group()

                    # Email pattern is very reliable - higher confidence
                    # National IDs (SSN, NINO, etc.) are also reliable patterns
                    if pattern_name == "email":
                        pattern_confidence = 0.90
                    elif pattern_name.startswith(("ssn_", "nino_", "insee_", "cf_", "dni_")):
                        pattern_confidence = 0.88
                    else:
                        pattern_confidence = min(confidence * 0.7, 0.75)

                    detected.append(
                        {
                            "block_id": block_id,
                            "text": matched_text,  # Use matched text, not entire block
                            "original_text": matched_text,
                            "full_block_text": text,  # Keep original for context
                            "bbox": bbox,
                            "field_type": pattern_name,
                            "confidence": pattern_confidence,
                            "page": page_num,
                            "detection_method": "pattern_match",
                            "regulations": PATTERN_REGULATIONS.get(pattern_name, ["GDPR"]),
                            "risk_level": PATTERN_RISK_LEVELS.get(pattern_name, "MEDIUM"),
                            "font_properties": block.get("font_properties", {}),
                        }
                    )
                    # Don't break - continue to find other patterns in same block

        # Legal entity detection
        legal_fields = self._detect_legal_entities(ocr_results, page_num)
        detected.extend(legal_fields)

        return detected

    def _is_likely_false_positive(self, text: str, pattern_name: str) -> bool:
        """Check if detection is likely a false positive."""
        # Date patterns are too generic - ignore by default in pattern matching
        # Dates should only be masked when LLM identifies them as DOB or sensitive
        if pattern_name in ("date_iso", "date_us", "date_eu"):
            return True

        return False

    def _detect_legal_entities(self, ocr_results: List[Dict], page_num: int) -> List[Dict]:
        """Detect company names by legal entity suffixes."""
        detected = []

        for block in ocr_results:
            text = block.get("text", "")
            bbox = block.get("bbox")
            block_id = block.get("block_id")
            confidence = block.get("confidence", 0.5)

            if not text or len(text) < 5:
                continue

            if has_legal_suffix(text):
                detected.append(
                    {
                        "block_id": block_id,
                        "text": text,
                        "original_text": text,
                        "bbox": bbox,
                        "field_type": "company_name",
                        "confidence": min(confidence + 0.1, 0.85),
                        "page": page_num,
                        "detection_method": "legal_suffix",
                        "regulations": ["GDPR", "KVKK"],  # Company names are PII
                        "risk_level": "MEDIUM",
                        "font_properties": block.get("font_properties", {}),
                    }
                )

        return detected

    def _deduplicate_fields(self, fields: List[Dict]) -> List[Dict]:
        """Remove duplicate fields based on bbox overlap and field type."""
        if not fields:
            return []

        # Sort by confidence
        sorted_fields = sorted(fields, key=lambda x: x.get("confidence", 0), reverse=True)

        unique = []
        # Track (block_id, field_type) pairs to allow multiple types per block
        seen_block_type_pairs = set()

        for field in sorted_fields:
            block_id = field.get("block_id")
            field_type = field.get("field_type", "unknown")
            bbox = field.get("bbox")
            text = field.get("text", "")

            # Create unique key: block_id + field_type + text
            # This allows same block to have multiple different detections
            unique_key = (block_id, field_type, text.lower().strip() if text else "")

            # Skip if already seen this exact combination
            if unique_key in seen_block_type_pairs:
                continue

            if not bbox:
                seen_block_type_pairs.add(unique_key)
                unique.append(field)
                continue

            # Check for duplicate only if same field_type
            is_duplicate = False
            for existing in unique:
                existing_type = existing.get("field_type", "unknown")
                existing_bbox = existing.get("bbox")

                # Only consider duplicate if same field type AND high overlap
                if existing_type == field_type and existing_bbox:
                    if self._bbox_overlap(bbox, existing_bbox) > 0.7:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(field)
                seen_block_type_pairs.add(unique_key)

        return unique

    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate overlap ratio between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if area1 == 0 or area2 == 0:
            return 0

        return intersection / min(area1, area2)

    async def close(self) -> None:
        """Close resources."""
        if self._classifier:
            await self._classifier.close()
            self._classifier = None
