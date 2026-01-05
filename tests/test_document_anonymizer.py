"""Tests for document_anonymizer module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from document_anonymizer.document_anonymizer import DocumentAnonymizer


class TestDocumentAnonymizerInit:
    """Tests for DocumentAnonymizer initialization."""

    def test_default_initialization(self):
        """Should initialize with default config."""
        with patch.dict("os.environ", {}, clear=True):
            anonymizer = DocumentAnonymizer()
            assert anonymizer.config is not None
            assert "anonymization" in anonymizer.config
            assert "detection_rules" in anonymizer.config
            assert "masking_strategy" in anonymizer.config

    def test_initialization_with_llm_params(self):
        """Should accept LLM parameters."""
        anonymizer = DocumentAnonymizer(
            llm_api_key="test_key", llm_api_url="https://test.api.com", llm_model="test-model"
        )
        assert anonymizer.config["llm"]["api_key"] == "test_key"
        assert anonymizer.config["llm"]["api_url"] == "https://test.api.com"
        assert anonymizer.config["llm"]["model"] == "test-model"

    def test_initialization_with_secret_key(self):
        """Should accept secret key parameter."""
        anonymizer = DocumentAnonymizer(secret_key="my_secret")
        assert anonymizer.config["anonymization"]["secret_key"] == "my_secret"

    def test_env_var_fallback(self):
        """Should use environment variables as fallback."""
        with patch.dict(
            "os.environ",
            {
                "LLM_API_KEY": "env_key",
                "LLM_API_URL": "https://env.api.com",
                "LLM_MODEL_VISION": "env-model",
            },
        ):
            anonymizer = DocumentAnonymizer()
            assert anonymizer.config["llm"]["api_key"] == "env_key"
            assert anonymizer.config["llm"]["api_url"] == "https://env.api.com"
            assert anonymizer.config["llm"]["model"] == "env-model"

    def test_explicit_params_override_env(self):
        """Explicit params should override env vars."""
        with patch.dict("os.environ", {"LLM_API_KEY": "env_key"}):
            anonymizer = DocumentAnonymizer(llm_api_key="explicit_key")
            assert anonymizer.config["llm"]["api_key"] == "explicit_key"

    def test_concurrency_settings(self):
        """Should set concurrency from env."""
        with patch.dict("os.environ", {"MAX_CONCURRENT_PAGES": "16", "MAX_CONCURRENT_FILES": "8"}):
            anonymizer = DocumentAnonymizer()
            assert anonymizer.max_concurrent_pages == 16
            assert anonymizer.max_concurrent_files == 8


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_default_config(self):
        """Should load default config without file."""
        anonymizer = DocumentAnonymizer()
        config = anonymizer.config

        # Check default values
        assert config["ocr_settings"]["dpi"] == 300
        assert config["detection_rules"]["use_llm_classification"] is True
        assert config["masking_strategy"]["text_fields"]["padding"] == 5

    def test_load_config_from_file(self):
        """Should load and merge config from YAML file."""
        config_content = """
ocr_settings:
  dpi: 200
detection_rules:
  min_confidence: 0.7
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()

            anonymizer = DocumentAnonymizer(config_path=f.name)

            # Custom values should override
            assert anonymizer.config["ocr_settings"]["dpi"] == 200
            assert anonymizer.config["detection_rules"]["min_confidence"] == 0.7
            # Defaults should remain
            assert anonymizer.config["verification"]["enabled"] is True

    def test_load_config_missing_file(self):
        """Should use defaults when file doesn't exist."""
        anonymizer = DocumentAnonymizer(config_path="/nonexistent/config.yaml")
        assert anonymizer.config["ocr_settings"]["dpi"] == 300


class TestDeepMerge:
    """Tests for _deep_merge helper."""

    @pytest.fixture
    def anonymizer(self):
        return DocumentAnonymizer()

    def test_simple_merge(self, anonymizer):
        """Should merge simple values."""
        base = {"a": 1, "b": 2}
        update = {"b": 3, "c": 4}
        anonymizer._deep_merge(base, update)

        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self, anonymizer):
        """Should merge nested dicts."""
        base = {"outer": {"inner1": 1, "inner2": 2}}
        update = {"outer": {"inner2": 3, "inner3": 4}}
        anonymizer._deep_merge(base, update)

        assert base["outer"]["inner1"] == 1
        assert base["outer"]["inner2"] == 3
        assert base["outer"]["inner3"] == 4

    def test_overwrite_non_dict(self, anonymizer):
        """Should overwrite when types differ."""
        base = {"key": "string"}
        update = {"key": {"nested": "dict"}}
        anonymizer._deep_merge(base, update)

        assert base["key"] == {"nested": "dict"}


class TestConfigureOutputPaths:
    """Tests for output path configuration."""

    @pytest.fixture
    def anonymizer(self):
        return DocumentAnonymizer()

    def test_creates_directories(self, anonymizer):
        """Should create output directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            anonymizer._configure_output_paths(str(output_dir))

            assert output_dir.exists()
            assert (output_dir / "logs").exists()
            assert (output_dir / "logs" / "reports").exists()

    def test_sets_registry_path(self, anonymizer):
        """Should set token registry path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            anonymizer._configure_output_paths(tmpdir)

            expected_path = str(Path(tmpdir) / "token_registry.json")
            assert anonymizer.config["anonymization"]["registry_path"] == expected_path


class TestAnonymizeDocument:
    """Tests for document anonymization."""

    @pytest.fixture
    def anonymizer(self):
        return DocumentAnonymizer()

    @pytest.mark.asyncio
    async def test_file_not_found(self, anonymizer):
        """Should return error report for missing file."""
        result = await anonymizer.anonymize_document("/nonexistent/file.pdf", "/tmp/output")

        assert result["status"] == "failed"
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, anonymizer):
        """Should return analysis without masking in dry run."""
        sample_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake PDF
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.touch()

            with (
                patch("document_anonymizer.document_anonymizer.get_pdf_info") as mock_info,
                patch("document_anonymizer.document_anonymizer.pdf_to_images") as mock_convert,
                patch.object(
                    anonymizer.field_detector, "detect_document_language", new_callable=AsyncMock
                ) as mock_lang,
                patch.object(
                    anonymizer.ocr_processor, "run_ocr", new_callable=AsyncMock
                ) as mock_ocr,
                patch.object(
                    anonymizer.field_detector, "detect_sensitive_fields", new_callable=AsyncMock
                ) as mock_detect,
            ):
                mock_info.return_value = {"page_count": 1}
                mock_convert.return_value = [sample_image]
                mock_lang.return_value = {"languages": ["en"], "locale": "en_US"}
                mock_ocr.return_value = {"text_blocks": []}
                mock_detect.return_value = ([], [])  # (auto_mask, needs_review)

                result = await anonymizer.anonymize_document(str(pdf_path), tmpdir, dry_run=True)

                assert result["status"] == "dry_run"
                assert result["total_pages"] == 1
                assert "detected_fields" in result

    @pytest.mark.asyncio
    async def test_failed_pdf_conversion(self, anonymizer):
        """Should return error when PDF conversion fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.touch()

            with (
                patch("document_anonymizer.document_anonymizer.get_pdf_info") as mock_info,
                patch("document_anonymizer.document_anonymizer.pdf_to_images") as mock_convert,
            ):
                mock_info.return_value = {"page_count": 1}
                mock_convert.return_value = []  # Empty - conversion failed

                result = await anonymizer.anonymize_document(str(pdf_path), tmpdir)

                assert result["status"] == "failed"
                assert "Failed to convert PDF" in result["error"]


class TestAnalyzeDocument:
    """Tests for analyze_document wrapper."""

    @pytest.fixture
    def anonymizer(self):
        return DocumentAnonymizer()

    @pytest.mark.asyncio
    async def test_analyze_calls_dry_run(self, anonymizer):
        """analyze_document should call anonymize_document with dry_run=True."""
        with patch.object(anonymizer, "anonymize_document", new_callable=AsyncMock) as mock_anon:
            mock_anon.return_value = {"status": "dry_run"}

            result = await anonymizer.analyze_document("/path/to/file.pdf")

            mock_anon.assert_called_once()
            call_args = mock_anon.call_args
            assert call_args[1]["dry_run"] is True


class TestAnonymizeBatch:
    """Tests for batch processing."""

    @pytest.fixture
    def anonymizer(self):
        return DocumentAnonymizer()

    @pytest.mark.asyncio
    async def test_batch_not_directory(self, anonymizer):
        """Should return empty list for non-directory."""
        result = await anonymizer.anonymize_batch("/nonexistent/folder", "/tmp/output")
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_empty_directory(self, anonymizer):
        """Should return empty list for directory with no PDFs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await anonymizer.anonymize_batch(tmpdir, tmpdir)
            assert result == []

    @pytest.mark.asyncio
    async def test_batch_processes_all_pdfs(self, anonymizer):
        """Should process all PDF files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake PDFs
            for i in range(3):
                (Path(tmpdir) / f"test{i}.pdf").touch()

            output_dir = Path(tmpdir) / "output"

            with patch.object(
                anonymizer, "anonymize_document", new_callable=AsyncMock
            ) as mock_anon:
                mock_anon.return_value = {
                    "status": "success",
                    "total_pages": 1,
                    "processing_time_seconds": 1.0,
                    "statistics": {"total_masked_fields": 0, "by_type": {}},
                }

                result = await anonymizer.anonymize_batch(tmpdir, str(output_dir))

                assert len(result) == 3
                assert mock_anon.call_count == 3


class TestTokenRegistry:
    """Tests for token registry management."""

    @pytest.fixture
    def anonymizer(self):
        return DocumentAnonymizer()

    def test_save_token_registry(self, anonymizer):
        """Should delegate to anonymization engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            anonymizer.save_token_registry(str(path))
            assert path.exists()

    def test_load_token_registry(self, anonymizer):
        """Should delegate to anonymization engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            # Create a valid registry file
            path.write_text("{}")

            # Should not raise
            anonymizer.load_token_registry(str(path))


class TestGetStatistics:
    """Tests for statistics retrieval."""

    def test_get_statistics(self):
        """Should return statistics from engine."""
        anonymizer = DocumentAnonymizer()
        stats = anonymizer.get_statistics()

        assert "anonymization_engine" in stats
        assert isinstance(stats["anonymization_engine"], dict)


class TestClose:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_cleanup(self):
        """Should close all resources."""
        anonymizer = DocumentAnonymizer()

        with (
            patch.object(
                anonymizer.field_detector, "close", new_callable=AsyncMock
            ) as mock_detector_close,
            patch.object(anonymizer.ocr_processor, "shutdown") as mock_ocr_shutdown,
        ):
            await anonymizer.close()

            mock_detector_close.assert_called_once()
            mock_ocr_shutdown.assert_called_once()


class TestReviewCallback:
    """Tests for review callback functionality."""

    @pytest.fixture
    def anonymizer(self):
        return DocumentAnonymizer()

    @pytest.mark.asyncio
    async def test_auto_approve_when_no_callback(self, anonymizer):
        """Should auto-approve fields when no callback provided."""
        sample_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.touch()

            needs_review_field = {
                "text": "John Smith",
                "field_type": "person_name",
                "page": 1,
                "bbox": (10, 10, 100, 30),
                "confidence": 0.75,
                "detection_method": "llm",
            }

            with (
                patch("document_anonymizer.document_anonymizer.get_pdf_info") as mock_info,
                patch("document_anonymizer.document_anonymizer.pdf_to_images") as mock_convert,
                patch("document_anonymizer.document_anonymizer.images_to_pdf") as mock_save,
                patch.object(
                    anonymizer.field_detector, "detect_document_language", new_callable=AsyncMock
                ) as mock_lang,
                patch.object(
                    anonymizer.ocr_processor, "run_ocr", new_callable=AsyncMock
                ) as mock_ocr,
                patch.object(
                    anonymizer.field_detector, "detect_sensitive_fields", new_callable=AsyncMock
                ) as mock_detect,
                patch.object(
                    anonymizer.verifier, "verify_masked_document", new_callable=AsyncMock
                ) as mock_verify,
            ):
                mock_info.return_value = {"page_count": 1}
                mock_convert.return_value = [sample_image]
                mock_lang.return_value = {"languages": ["en"], "locale": "en_US"}
                mock_ocr.return_value = {"text_blocks": []}
                mock_detect.return_value = (
                    [],
                    [needs_review_field],
                )  # auto_mask empty, needs_review has 1

                # Mock verification
                from document_anonymizer.verification import VerificationResult, VerificationStatus

                mock_verify.return_value = VerificationResult(
                    status=VerificationStatus.PASSED, confidence_score=0.95
                )

                result = await anonymizer.anonymize_document(
                    str(pdf_path), tmpdir, auto_approve_unreviewed=True
                )

                # Field should be included (auto-approved)
                assert result["status"] == "success"
