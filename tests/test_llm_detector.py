"""Tests for llm_classifier module (LLMDetector)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from document_anonymizer.llm_classifier import LLMDetector


class TestLLMDetectorInit:
    """Tests for LLMDetector initialization."""

    def test_default_initialization(self):
        """Should initialize with defaults."""
        detector = LLMDetector()
        assert detector.config == {}
        assert detector.max_retries == 3
        assert detector.timeout == 120
        assert detector.retry_delay == 2

    def test_env_var_configuration(self):
        """Should use environment variables."""
        with patch.dict(
            "os.environ",
            {
                "LLM_API_URL": "https://test.api.com",
                "LLM_API_KEY": "test_key",
                "LLM_MODEL_VISION": "test-model",
            },
        ):
            detector = LLMDetector()
            assert detector.api_url == "https://test.api.com"
            assert detector.api_key == "test_key"
            assert detector.model == "test-model"

    def test_config_override(self):
        """Config should override environment variables."""
        config = {
            "llm": {
                "api_url": "https://config.api.com",
                "api_key": "config_key",
                "model": "config-model",
                "max_retries": 5,
                "timeout": 60,
            }
        }
        detector = LLMDetector(config=config)
        assert detector.api_url == "https://config.api.com"
        assert detector.api_key == "config_key"
        assert detector.model == "config-model"
        assert detector.max_retries == 5
        assert detector.timeout == 60

    def test_prompt_loading(self):
        """Should load prompt from file."""
        detector = LLMDetector()
        # Prompt should be loaded (non-empty if file exists)
        assert isinstance(detector._prompt, str)


class TestImageEncoding:
    """Tests for image encoding."""

    @pytest.fixture
    def detector(self):
        return LLMDetector()

    def test_encode_image_basic(self, detector):
        """Should encode image to base64."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = detector._encode_image(img)

        assert isinstance(result, str)
        # Should be valid base64
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0


class TestDetectDocumentLanguage:
    """Tests for language detection."""

    @pytest.fixture
    def detector(self):
        return LLMDetector()

    @pytest.mark.asyncio
    async def test_detect_language_no_api_key(self, detector):
        """Should return defaults when no API key."""
        detector.api_key = ""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = await detector.detect_document_language(img)

        assert result == {"languages": ["en"], "locale": "en_US"}

    @pytest.mark.asyncio
    async def test_detect_language_with_mock(self, detector):
        """Should parse LLM response correctly."""
        detector.api_key = "test_key"

        mock_response = '{"languages": ["en", "tr"], "locale": "tr_TR"}'

        with patch.object(detector, "_call_llm_with_image", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255

            result = await detector.detect_document_language(img)

            assert result["languages"] == ["en", "tr"]
            assert result["locale"] == "tr_TR"

    @pytest.mark.asyncio
    async def test_detect_language_ensures_english(self, detector):
        """Should ensure 'en' is always in languages."""
        detector.api_key = "test_key"

        # Response without 'en'
        mock_response = '{"languages": ["tr"], "locale": "tr_TR"}'

        with patch.object(detector, "_call_llm_with_image", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255

            result = await detector.detect_document_language(img)

            assert "en" in result["languages"]


class TestDetectAll:
    """Tests for unified detection."""

    @pytest.fixture
    def detector(self):
        detector = LLMDetector()
        detector.api_key = "test_key"
        detector._prompt = "Test prompt"
        return detector

    @pytest.mark.asyncio
    async def test_detect_all_no_api_key(self):
        """Should return empty result when no API key."""
        detector = LLMDetector()
        detector.api_key = ""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = await detector.detect_all(img)

        assert result == {"text_detections": [], "visual_detections": []}

    @pytest.mark.asyncio
    async def test_detect_all_no_prompt(self, detector):
        """Should return empty result when no prompt."""
        detector._prompt = ""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = await detector.detect_all(img)

        assert result == {"text_detections": [], "visual_detections": []}

    @pytest.mark.asyncio
    async def test_detect_all_parses_response(self, detector):
        """Should parse LLM response correctly."""
        mock_response = json.dumps(
            {
                "text_detections": [
                    {
                        "block_id": "block_0",
                        "full_text": "Name: John Smith",
                        "label": "Name: ",
                        "sensitive_value": "John Smith",
                        "category": "person_name",
                        "confidence": 0.95,
                        "risk_level": "HIGH",
                    }
                ],
                "visual_detections": [
                    {
                        "element_id": "sig_1",
                        "type": "signature",
                        "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 250},
                        "confidence": 0.9,
                    }
                ],
            }
        )

        ocr_results = [
            {"block_id": "block_0", "text": "Name: John Smith", "bbox": [10, 20, 100, 40]}
        ]

        with patch.object(detector, "_call_llm_with_image", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255

            result = await detector.detect_all(img, ocr_results)

            assert len(result["text_detections"]) == 1
            assert result["text_detections"][0]["field_type"] == "person_name"
            assert result["text_detections"][0]["label"] == "Name: "
            assert result["text_detections"][0]["sensitive_value"] == "John Smith"

            assert len(result["visual_detections"]) == 1
            assert result["visual_detections"][0]["field_type"] == "signature"


class TestParseUnifiedResponse:
    """Tests for response parsing."""

    @pytest.fixture
    def detector(self):
        return LLMDetector()

    def test_parse_empty_response(self, detector):
        """Should handle empty response."""
        result = detector._parse_unified_response("", [])
        assert result == {"text_detections": [], "visual_detections": []}

    def test_parse_invalid_json(self, detector):
        """Should handle invalid JSON."""
        result = detector._parse_unified_response("not json", [])
        assert result == {"text_detections": [], "visual_detections": []}

    def test_parse_valid_response(self, detector):
        """Should parse valid response."""
        response = json.dumps(
            {
                "text_detections": [
                    {"block_id": "b1", "sensitive_value": "test", "category": "email"}
                ],
                "visual_detections": [],
            }
        )
        ocr_results = [{"block_id": "b1", "bbox": [0, 0, 100, 20]}]

        result = detector._parse_unified_response(response, ocr_results)

        assert len(result["text_detections"]) == 1
        assert result["text_detections"][0]["bbox"] == [0, 0, 100, 20]


class TestClientManagement:
    """Tests for HTTP client management."""

    @pytest.fixture
    def detector(self):
        return LLMDetector()

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, detector):
        """Should create HTTP client on first call."""
        assert detector._client is None
        client = await detector._get_client()
        assert client is not None
        assert detector._client is client

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, detector):
        """Should reuse existing client."""
        client1 = await detector._get_client()
        client2 = await detector._get_client()
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_close_client(self, detector):
        """Should close client properly."""
        await detector._get_client()
        assert detector._client is not None

        await detector.close()
        assert detector._client is None


class TestPromptLoading:
    """Tests for prompt file loading."""

    def test_prompt_loaded_from_file(self):
        """Should load prompt from file."""
        detector = LLMDetector()
        # Prompt should be loaded (non-empty if file exists)
        assert isinstance(detector._prompt, str)


class TestDetectAllEdgeCases:
    """Tests for detect_all edge cases."""

    @pytest.fixture
    def detector(self):
        detector = LLMDetector()
        detector.api_key = "test_key"
        detector._prompt = "Test prompt"
        return detector

    @pytest.mark.asyncio
    async def test_detect_all_with_empty_response(self, detector):
        """Should handle empty LLM response."""
        with patch.object(detector, "_call_llm_with_image", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ""
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255

            result = await detector.detect_all(img)

            assert result == {"text_detections": [], "visual_detections": []}

    @pytest.mark.asyncio
    async def test_detect_all_with_ocr_results(self, detector):
        """Should use OCR results for bbox lookup."""
        mock_response = json.dumps(
            {
                "text_detections": [
                    {
                        "block_id": "block_1",
                        "sensitive_value": "test@email.com",
                        "category": "email",
                        "confidence": 0.9,
                    }
                ],
                "visual_detections": [],
            }
        )

        ocr_results = [
            {"block_id": "block_1", "text": "Email: test@email.com", "bbox": [10, 20, 200, 40]}
        ]

        with patch.object(detector, "_call_llm_with_image", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255

            result = await detector.detect_all(img, ocr_results)

            assert len(result["text_detections"]) == 1
            assert result["text_detections"][0]["bbox"] == [10, 20, 200, 40]


class TestParseResponseEdgeCases:
    """Tests for response parsing edge cases."""

    @pytest.fixture
    def detector(self):
        return LLMDetector()

    def test_parse_response_with_markdown(self, detector):
        """Should handle markdown-wrapped JSON."""
        response = '```json\n{"text_detections": [], "visual_detections": []}\n```'
        result = detector._parse_unified_response(response, [])
        assert "text_detections" in result

    def test_parse_response_missing_fields(self, detector):
        """Should handle missing required fields."""
        response = json.dumps(
            {
                "text_detections": [{"block_id": "b1"}],  # Missing other fields
                "visual_detections": [],
            }
        )
        result = detector._parse_unified_response(response, [])
        # Should not crash, may have empty or partial results
        assert isinstance(result, dict)

    def test_parse_visual_detection_bbox(self, detector):
        """Should parse visual detection bbox correctly."""
        response = json.dumps(
            {
                "text_detections": [],
                "visual_detections": [
                    {
                        "element_id": "sig_1",
                        "type": "signature",
                        "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 250},
                        "confidence": 0.9,
                    }
                ],
            }
        )
        result = detector._parse_unified_response(response, [])

        assert len(result["visual_detections"]) == 1
        assert result["visual_detections"][0]["bbox"] == [100, 200, 300, 250]


class TestLanguageDetectionEdgeCases:
    """Tests for language detection edge cases."""

    @pytest.fixture
    def detector(self):
        detector = LLMDetector()
        detector.api_key = "test_key"
        return detector

    @pytest.mark.asyncio
    async def test_detect_language_with_invalid_json(self, detector):
        """Should handle invalid JSON response."""
        with patch.object(detector, "_call_llm_with_image", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "not valid json"
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255

            result = await detector.detect_document_language(img)

            # Should return defaults on error
            assert "languages" in result
            assert "locale" in result

    @pytest.mark.asyncio
    async def test_detect_language_api_error(self, detector):
        """Should handle API errors gracefully."""
        with patch.object(detector, "_call_llm_with_image", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API Error")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255

            result = await detector.detect_document_language(img)

            # Should return defaults on error
            assert result == {"languages": ["en"], "locale": "en_US"}
