"""Tests for field_detector module."""

import pytest
import numpy as np

from document_anonymizer.field_detector import (
    FieldDetector,
    CONFIDENCE_AUTO_MASK,
    CONFIDENCE_REVIEW,
    PATTERN_REGULATIONS,
    PATTERN_RISK_LEVELS,
)


class TestFieldDetectorConstants:
    """Tests for field detector constants."""

    def test_confidence_thresholds(self):
        """Confidence thresholds should be reasonable."""
        assert CONFIDENCE_AUTO_MASK > CONFIDENCE_REVIEW
        assert CONFIDENCE_AUTO_MASK <= 1.0
        assert CONFIDENCE_REVIEW >= 0.0

    def test_pattern_regulations_mapping(self):
        """Pattern regulations should have valid values."""
        for pattern_type, regulations in PATTERN_REGULATIONS.items():
            assert isinstance(regulations, list)
            assert len(regulations) > 0
            # Check for known regulations
            valid_regs = {"GDPR", "KVKK", "CCPA", "LGPD", "HIPAA"}
            for reg in regulations:
                assert reg in valid_regs, f"Unknown regulation: {reg}"

    def test_pattern_risk_levels(self):
        """Pattern risk levels should be valid."""
        valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for pattern_type, level in PATTERN_RISK_LEVELS.items():
            assert level in valid_levels, f"Invalid risk level: {level}"


class TestFieldDetectorInit:
    """Tests for FieldDetector initialization."""

    def test_default_initialization(self):
        """Should initialize with default config."""
        detector = FieldDetector()
        assert detector.use_llm_classification is True
        assert detector.use_fallback_detection is True
        assert detector.min_confidence == CONFIDENCE_REVIEW
        assert detector.auto_mask_confidence == CONFIDENCE_AUTO_MASK

    def test_custom_config(self):
        """Should respect custom configuration."""
        config = {
            "detection_rules": {
                "use_llm_classification": False,
                "use_fallback_detection": False,
                "min_confidence": 0.5,
                "auto_mask_confidence": 0.9,
            }
        }
        detector = FieldDetector(config=config)

        assert detector.use_llm_classification is False
        assert detector.use_fallback_detection is False
        assert detector.min_confidence == 0.5
        assert detector.auto_mask_confidence == 0.9

    def test_patterns_compiled(self):
        """Patterns should be compiled on init."""
        detector = FieldDetector()
        # Should have compiled patterns
        assert len(detector._patterns) > 0

    def test_lazy_classifier_loading(self):
        """Classifier should be None until first use."""
        detector = FieldDetector()
        assert detector._classifier is None


class TestFieldDetectorPatternMatching:
    """Tests for pattern-based field detection."""

    @pytest.fixture
    def detector(self):
        """Create detector with LLM disabled."""
        return FieldDetector(
            config={
                "detection_rules": {
                    "use_llm_classification": False,
                    "use_fallback_detection": True,
                }
            }
        )

    def test_detect_email_pattern(self, detector):
        """Should detect email patterns."""
        # Test email regex directly
        email_pattern = detector._patterns.get("email")
        if email_pattern:
            assert email_pattern.search("john@example.com") is not None
            assert email_pattern.search("test.user@company.co.uk") is not None

    def test_detect_phone_pattern(self, detector):
        """Should detect country-specific phone patterns."""
        # Test Turkish phone pattern
        phone_tr = detector._patterns.get("phone_tr")
        if phone_tr:
            assert phone_tr.search("+90 532 123 45 67") is not None

    def test_detect_ip_address_pattern(self, detector):
        """Should detect IP address patterns."""
        ip_pattern = detector._patterns.get("ip_address")
        if ip_pattern:
            assert ip_pattern.search("192.168.1.1") is not None
            assert ip_pattern.search("10.0.0.1") is not None


class TestFieldDetectorSeparation:
    """Tests for field separation by confidence."""

    @pytest.fixture
    def detector(self):
        return FieldDetector()

    def test_separate_by_confidence(self, detector):
        """Fields should be separated by confidence threshold."""
        fields = [
            {"text": "high", "confidence": 0.95},
            {"text": "medium", "confidence": 0.75},
            {"text": "low", "confidence": 0.50},
        ]

        auto_mask = [f for f in fields if f["confidence"] >= detector.auto_mask_confidence]
        review = [
            f
            for f in fields
            if detector.min_confidence <= f["confidence"] < detector.auto_mask_confidence
        ]
        ignored = [f for f in fields if f["confidence"] < detector.min_confidence]

        assert len(auto_mask) == 1
        assert auto_mask[0]["text"] == "high"
        assert len(review) == 1
        assert review[0]["text"] == "medium"
        assert len(ignored) == 1


class TestFieldDetectorHelpers:
    """Tests for helper methods."""

    @pytest.fixture
    def detector(self):
        return FieldDetector()

    def test_pattern_to_field_type_mapping(self, detector):
        """Pattern names should map to field types correctly."""
        # Common patterns should exist
        assert "email" in detector._patterns
        assert "ip_address" in detector._patterns


class TestAsyncDetection:
    """Tests for async detection methods."""

    @pytest.fixture
    def detector(self):
        """Detector with LLM disabled for testing."""
        return FieldDetector(
            config={
                "detection_rules": {
                    "use_llm_classification": False,
                    "use_fallback_detection": True,
                }
            }
        )

    @pytest.fixture
    def sample_image(self):
        """Sample image for testing."""
        return np.ones((100, 100, 3), dtype=np.uint8) * 255

    @pytest.fixture
    def sample_ocr_results(self):
        """Sample OCR results."""
        return [
            {
                "text": "john@example.com",
                "bbox": [[10, 10], [90, 10], [90, 20], [10, 20]],
                "confidence": 0.95,
                "block_id": 1,
            },
            {
                "text": "+90 532 123 4567",
                "bbox": [[10, 30], [90, 30], [90, 40], [10, 40]],
                "confidence": 0.90,
                "block_id": 2,
            },
        ]

    @pytest.mark.asyncio
    async def test_detect_sensitive_fields_returns_tuple(
        self, detector, sample_image, sample_ocr_results
    ):
        """detect_sensitive_fields should return tuple of two lists."""
        result = await detector.detect_sensitive_fields(
            sample_image, sample_ocr_results, page_num=1
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # auto_mask
        assert isinstance(result[1], list)  # review


class TestRiskLevelMapping:
    """Tests for risk level assignment."""

    def test_critical_patterns(self):
        """Critical patterns should be correctly identified."""
        critical_patterns = [k for k, v in PATTERN_RISK_LEVELS.items() if v == "CRITICAL"]
        assert "ssn_us" in critical_patterns
        assert "nino_uk" in critical_patterns
        assert "cf_it" in critical_patterns

    def test_high_risk_patterns(self):
        """High risk patterns should be correctly identified."""
        high_patterns = [k for k, v in PATTERN_RISK_LEVELS.items() if v == "HIGH"]
        assert "plate_tr" in high_patterns
        assert "plate_uk" in high_patterns
        assert "plate_fr" in high_patterns

    def test_medium_risk_patterns(self):
        """Medium risk patterns should be correctly identified."""
        medium_patterns = [k for k, v in PATTERN_RISK_LEVELS.items() if v == "MEDIUM"]
        assert "email" in medium_patterns
        assert "phone_tr" in medium_patterns
        assert "ip_address" in medium_patterns

    def test_low_risk_patterns(self):
        """Low risk patterns should be correctly identified."""
        low_patterns = [k for k, v in PATTERN_RISK_LEVELS.items() if v == "LOW"]
        assert "date_iso" in low_patterns or "date_eu" in low_patterns
        assert "postal_uk" in low_patterns


class TestRegulationMapping:
    """Tests for regulation mapping."""

    def test_gdpr_coverage(self):
        """GDPR should cover many patterns."""
        gdpr_patterns = [k for k, v in PATTERN_REGULATIONS.items() if "GDPR" in v]
        assert len(gdpr_patterns) >= 3

    def test_kvkk_coverage(self):
        """KVKK should cover relevant patterns."""
        kvkk_patterns = [k for k, v in PATTERN_REGULATIONS.items() if "KVKK" in v]
        assert "phone_tr" in kvkk_patterns
        assert "plate_tr" in kvkk_patterns

    def test_ccpa_coverage(self):
        """CCPA should cover relevant patterns."""
        ccpa_patterns = [k for k, v in PATTERN_REGULATIONS.items() if "CCPA" in v]
        assert "ssn_us" in ccpa_patterns


class TestPatternDetection:
    """Tests for _detect_by_patterns method."""

    @pytest.fixture
    def detector(self):
        return FieldDetector(
            config={
                "detection_rules": {
                    "use_llm_classification": False,
                    "use_fallback_detection": True,
                }
            }
        )

    def test_detect_email_in_ocr_results(self, detector):
        """Should detect email in OCR results."""
        ocr_results = [
            {
                "text": "Contact: john@example.com",
                "bbox": [0, 0, 100, 20],
                "block_id": "b1",
                "confidence": 0.9,
            }
        ]
        fields = detector._detect_by_patterns(ocr_results, page_num=1)

        email_fields = [f for f in fields if f["field_type"] == "email"]
        assert len(email_fields) >= 1
        assert email_fields[0]["text"] == "john@example.com"

    def test_detect_phone_in_ocr_results(self, detector):
        """Should detect phone numbers."""
        ocr_results = [
            {
                "text": "Tel: +90 532 123 4567",
                "bbox": [0, 0, 100, 20],
                "block_id": "b1",
                "confidence": 0.9,
            }
        ]
        fields = detector._detect_by_patterns(ocr_results, page_num=1)

        phone_fields = [f for f in fields if "phone" in f["field_type"]]
        assert len(phone_fields) >= 1

    def test_skip_short_text(self, detector):
        """Should skip text shorter than 3 characters."""
        ocr_results = [{"text": "AB", "bbox": [0, 0, 20, 10], "block_id": "b1", "confidence": 0.9}]
        fields = detector._detect_by_patterns(ocr_results, page_num=1)
        assert len(fields) == 0


class TestFalsePositiveDetection:
    """Tests for false positive filtering."""

    @pytest.fixture
    def detector(self):
        return FieldDetector()

    def test_date_is_false_positive(self, detector):
        """Date patterns should be filtered as false positive."""
        # Dates are generic and should only be masked when LLM identifies them
        assert detector._is_likely_false_positive("2024-01-15", "date_iso") is True
        assert detector._is_likely_false_positive("01/15/2024", "date_us") is True

    def test_email_is_not_false_positive(self, detector):
        """Email should not be filtered."""
        assert detector._is_likely_false_positive("john@example.com", "email") is False


class TestLegalEntityDetection:
    """Tests for legal entity detection."""

    @pytest.fixture
    def detector(self):
        return FieldDetector()

    def test_detect_ltd_company(self, detector):
        """Should detect Ltd company."""
        ocr_results = [
            {
                "text": "Acme Technologies Ltd.",
                "bbox": [0, 0, 200, 20],
                "block_id": "b1",
                "confidence": 0.9,
            }
        ]
        fields = detector._detect_legal_entities(ocr_results, page_num=1)

        assert len(fields) >= 1
        assert fields[0]["field_type"] == "company_name"

    def test_detect_gmbh_company(self, detector):
        """Should detect GmbH company."""
        ocr_results = [
            {
                "text": "Deutsche Software GmbH",
                "bbox": [0, 0, 200, 20],
                "block_id": "b1",
                "confidence": 0.9,
            }
        ]
        fields = detector._detect_legal_entities(ocr_results, page_num=1)

        assert len(fields) >= 1
        assert fields[0]["field_type"] == "company_name"

    def test_detect_as_company(self, detector):
        """Should detect A.Ş. (Turkish) company."""
        ocr_results = [
            {
                "text": "Turk Telekom A.Ş.",
                "bbox": [0, 0, 200, 20],
                "block_id": "b1",
                "confidence": 0.9,
            }
        ]
        fields = detector._detect_legal_entities(ocr_results, page_num=1)

        assert len(fields) >= 1

    def test_skip_short_company_names(self, detector):
        """Should skip short texts."""
        ocr_results = [{"text": "Ltd", "bbox": [0, 0, 30, 20], "block_id": "b1", "confidence": 0.9}]
        fields = detector._detect_legal_entities(ocr_results, page_num=1)
        # Less than 5 chars should be skipped
        assert len(fields) == 0


class TestDeduplication:
    """Tests for field deduplication."""

    @pytest.fixture
    def detector(self):
        return FieldDetector()

    def test_remove_exact_duplicates(self, detector):
        """Should remove exact duplicates."""
        fields = [
            {
                "block_id": "b1",
                "field_type": "email",
                "text": "a@b.com",
                "confidence": 0.9,
                "bbox": [0, 0, 100, 20],
            },
            {
                "block_id": "b1",
                "field_type": "email",
                "text": "a@b.com",
                "confidence": 0.8,
                "bbox": [0, 0, 100, 20],
            },
        ]

        unique = detector._deduplicate_fields(fields)
        assert len(unique) == 1
        # Higher confidence should be kept
        assert unique[0]["confidence"] == 0.9

    def test_keep_different_types(self, detector):
        """Should keep fields of different types from same block."""
        fields = [
            {
                "block_id": "b1",
                "field_type": "email",
                "text": "a@b.com",
                "confidence": 0.9,
                "bbox": [0, 0, 100, 20],
            },
            {
                "block_id": "b1",
                "field_type": "person_name",
                "text": "John",
                "confidence": 0.8,
                "bbox": [0, 0, 100, 20],
            },
        ]

        unique = detector._deduplicate_fields(fields)
        assert len(unique) == 2

    def test_empty_fields(self, detector):
        """Should handle empty list."""
        unique = detector._deduplicate_fields([])
        assert unique == []


class TestBboxOverlap:
    """Tests for bbox overlap calculation."""

    @pytest.fixture
    def detector(self):
        return FieldDetector()

    def test_full_overlap(self, detector):
        """Same bbox should have 100% overlap."""
        bbox1 = [0, 0, 100, 50]
        bbox2 = [0, 0, 100, 50]
        overlap = detector._bbox_overlap(bbox1, bbox2)
        assert overlap == 1.0

    def test_no_overlap(self, detector):
        """Non-overlapping bboxes should have 0% overlap."""
        bbox1 = [0, 0, 50, 50]
        bbox2 = [100, 100, 150, 150]
        overlap = detector._bbox_overlap(bbox1, bbox2)
        assert overlap == 0.0

    def test_partial_overlap(self, detector):
        """Partially overlapping bboxes."""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [50, 50, 150, 150]
        overlap = detector._bbox_overlap(bbox1, bbox2)
        assert 0 < overlap < 1

    def test_zero_area_bbox(self, detector):
        """Zero-area bbox should return 0."""
        bbox1 = [0, 0, 0, 0]  # Zero width and height
        bbox2 = [0, 0, 100, 100]
        overlap = detector._bbox_overlap(bbox1, bbox2)
        assert overlap == 0


class TestAsyncClose:
    """Tests for async close method."""

    @pytest.mark.asyncio
    async def test_close_without_classifier(self):
        """Should close cleanly without classifier."""
        detector = FieldDetector()
        await detector.close()
        assert detector._classifier is None

    @pytest.mark.asyncio
    async def test_close_with_classifier(self):
        """Should close classifier if initialized."""
        detector = FieldDetector()
        # Manually set classifier to test close
        from unittest.mock import AsyncMock, MagicMock

        mock_classifier = MagicMock()
        mock_classifier.close = AsyncMock()
        detector._classifier = mock_classifier

        await detector.close()

        mock_classifier.close.assert_called_once()
        assert detector._classifier is None
