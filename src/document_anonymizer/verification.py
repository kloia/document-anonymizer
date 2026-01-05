"""Post-masking verification for sensitive data leakage detection."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .constants import (
    BOILERPLATE_PATTERNS,
    EXPECTED_MASKED_PATTERNS,
    GENERIC_DOCUMENT_PHRASES,
    VERIFICATION_SENSITIVE_PATTERNS,
)

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Verification result status."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VerificationResult:
    """Result of verification check."""

    status: VerificationStatus
    leaked_fields: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    pages_checked: int = 0
    confidence_score: float = 1.0
    details: Dict = field(default_factory=dict)


class PostMaskingVerifier:
    """Verifies masked documents for sensitive data leakage."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize verifier."""
        self.config = config or {}

        self._sensitive_patterns = {}
        for field_type, patterns in VERIFICATION_SENSITIVE_PATTERNS.items():
            self._sensitive_patterns[field_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

        self._expected_patterns = [re.compile(p) for p in EXPECTED_MASKED_PATTERNS]

        verification_config = self.config.get("verification", {})
        self.enabled = verification_config.get("enabled", True)
        self.strict_mode = verification_config.get("strict_mode", False)
        self.check_original_text = verification_config.get("check_original_text", True)

        logger.debug("Post-masking verifier initialized")

    async def verify_masked_document(
        self, masked_images: List[np.ndarray], original_fields: List[Dict], ocr_processor=None
    ) -> VerificationResult:
        """
        Verify that a masked document doesn't contain leaked sensitive data.

        Args:
            masked_images: List of masked page images
            original_fields: Original sensitive fields that were masked
            ocr_processor: OCRProcessor instance

        Returns:
            VerificationResult with status and details
        """
        if not self.enabled:
            return VerificationResult(
                status=VerificationStatus.SKIPPED, details={"reason": "Verification disabled"}
            )

        from .ocr_processor import OCRProcessor

        if ocr_processor is None:
            ocr_processor = OCRProcessor(self.config)

        leaked_fields = []
        warnings = []
        pages_checked = 0
        total_confidence = 1.0

        original_texts = set()
        for orig_field in original_fields:
            text = orig_field.get("original_text", "").strip()
            if text and len(text) >= 3:
                original_texts.add(text.lower())

        for page_num, image in enumerate(masked_images, start=1):
            pages_checked += 1

            try:
                ocr_result = await ocr_processor.run_ocr(image, page_num)
                text_blocks = ocr_result.get("text_blocks", [])
                full_text = ocr_result.get("full_text", "")

                page_leaks = self._check_for_patterns(full_text, page_num)
                leaked_fields.extend(page_leaks)

                if self.check_original_text:
                    original_leaks = self._check_for_original_text(
                        full_text, text_blocks, original_texts, page_num
                    )
                    leaked_fields.extend(original_leaks)

                if page_leaks:
                    total_confidence *= 0.7

            except Exception as e:
                logger.error(f"Verification error on page {page_num}: {e}")
                warnings.append(f"Page {page_num}: Verification error - {str(e)}")

        if not leaked_fields:
            status = VerificationStatus.PASSED
            confidence_score = total_confidence
        else:
            high_confidence_leaks = [f for f in leaked_fields if f.get("confidence", 0) > 0.8]

            if high_confidence_leaks:
                status = VerificationStatus.FAILED
                confidence_score = 0.0
            else:
                status = VerificationStatus.WARNING
                confidence_score = max(0.3, total_confidence * 0.5)

        return VerificationResult(
            status=status,
            leaked_fields=leaked_fields,
            warnings=warnings,
            pages_checked=pages_checked,
            confidence_score=confidence_score,
            details={
                "total_leaks_found": len(leaked_fields),
                "original_fields_count": len(original_fields),
                "pages_processed": pages_checked,
            },
        )

    def _check_for_patterns(self, text: str, page_num: int) -> List[Dict]:
        """Check text for sensitive patterns."""
        leaked = []

        for field_type, patterns in self._sensitive_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matched_text = match.group()

                    if self._is_expected_masked_text(matched_text):
                        continue

                    if len(matched_text) < 4:
                        continue

                    leaked.append(
                        {
                            "text": matched_text,
                            "field_type": field_type,
                            "page": page_num,
                            "confidence": 0.7,
                            "detection_method": "pattern_match",
                            "reason": f"Sensitive pattern ({field_type}) found",
                        }
                    )

        return leaked

    def _check_for_original_text(
        self, full_text: str, text_blocks: List[Dict], original_texts: set, page_num: int
    ) -> List[Dict]:
        """Check if original sensitive text is still present."""
        leaked = []
        full_text_lower = full_text.lower()

        boilerplate_compiled = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]

        for original in original_texts:
            if len(original) < 5:
                continue

            if original in GENERIC_DOCUMENT_PHRASES:
                continue

            is_boilerplate = False
            for pattern in boilerplate_compiled:
                if pattern.search(original):
                    is_boilerplate = True
                    break
            if is_boilerplate:
                continue

            if original in full_text_lower:
                for block in text_blocks:
                    block_text = block.get("text", "").lower()
                    if original in block_text:
                        leaked.append(
                            {
                                "text": original,
                                "field_type": "original_text_leaked",
                                "page": page_num,
                                "bbox": block.get("bbox"),
                                "confidence": 0.95,
                                "detection_method": "original_text_match",
                                "reason": "Original text found in masked document",
                            }
                        )
                        break

        return leaked

    def _is_expected_masked_text(self, text: str) -> bool:
        """Check if text matches expected dummy/masked patterns."""
        for pattern in self._expected_patterns:
            if pattern.search(text):
                return True
        return False

    def generate_verification_report(self, result: VerificationResult, document_name: str) -> Dict:
        """Generate a detailed verification report."""
        report = {
            "document": document_name,
            "verification_status": result.status.value,
            "confidence_score": round(result.confidence_score, 3),
            "pages_checked": result.pages_checked,
            "summary": {
                "total_leaked_fields": len(result.leaked_fields),
                "total_warnings": len(result.warnings),
            },
            "passed": result.status == VerificationStatus.PASSED,
        }

        if result.leaked_fields:
            report["leaked_fields"] = [
                {
                    "text": f["text"][:50] + "..."
                    if len(f.get("text", "")) > 50
                    else f.get("text", ""),
                    "field_type": f.get("field_type"),
                    "page": f.get("page"),
                    "confidence": f.get("confidence"),
                    "reason": f.get("reason"),
                }
                for f in result.leaked_fields
            ]

        if result.warnings:
            report["warnings"] = result.warnings

        if result.status == VerificationStatus.FAILED:
            report["recommendations"] = [
                "Review the listed leaked fields manually",
                "Re-process the document with adjusted settings",
            ]
        elif result.status == VerificationStatus.WARNING:
            report["recommendations"] = [
                "Review potential leaks for false positives",
            ]

        return report
