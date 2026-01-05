"""Tests for report_generator module."""

import json
import tempfile
from pathlib import Path

import pytest

from document_anonymizer.report_generator import ReportGenerator


class TestReportGeneratorInit:
    """Tests for ReportGenerator initialization."""

    def test_default_initialization(self):
        """Should initialize with defaults."""
        generator = ReportGenerator()
        assert generator.config == {}
        assert generator.output_dir is None

    def test_with_config(self):
        """Should accept config."""
        config = {"logging": {"report_dir": "custom/reports"}}
        generator = ReportGenerator(config=config)
        assert generator.config == config

    def test_with_output_dir(self):
        """Should accept output_dir."""
        generator = ReportGenerator(output_dir="/tmp/output")
        assert generator.output_dir == Path("/tmp/output")


class TestReportDir:
    """Tests for report directory handling."""

    def test_report_dir_from_output_dir(self):
        """Should create report dir under output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)
            report_dir = generator.report_dir

            assert report_dir == Path(tmpdir) / "logs" / "reports"
            assert report_dir.exists()

    def test_report_dir_from_config(self):
        """Should use config report_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"logging": {"report_dir": tmpdir}}
            generator = ReportGenerator(config=config)
            report_dir = generator.report_dir

            assert report_dir == Path(tmpdir)
            assert report_dir.exists()

    def test_set_output_dir(self):
        """Should update output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator()
            generator.set_output_dir(tmpdir)

            assert generator.output_dir == Path(tmpdir)
            # report_dir should be reset
            assert generator._report_dir is None


class TestGenerateReport:
    """Tests for report generation."""

    @pytest.fixture
    def generator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ReportGenerator(output_dir=tmpdir)

    @pytest.fixture
    def sample_fields(self):
        return [
            {
                "field_type": "person_name",
                "page": 1,
                "bbox": [10, 20, 100, 40],
                "confidence": 0.95,
                "original_text": "John Smith",
                "dummy_text": "Alex Johnson",
                "detection_method": "llm_unified",
                "regulations": ["GDPR", "KVKK"],
                "risk_level": "HIGH",
            },
            {
                "field_type": "email",
                "page": 1,
                "bbox": [10, 50, 200, 70],
                "confidence": 0.90,
                "original_text": "john@example.com",
                "detection_method": "pattern_match",
            },
            {
                "field_type": "signature",
                "page": 2,
                "bbox": [100, 200, 300, 280],
                "confidence": 0.85,
                "detection_method": "visual_detection",
            },
        ]

    def test_generate_report_basic(self, generator, sample_fields):
        """Should generate basic report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.pdf"
            input_path.touch()
            output_path.write_bytes(b"fake pdf content")

            report = generator.generate_report(
                input_path=str(input_path),
                output_path=str(output_path),
                detected_fields=sample_fields,
                hash_mapping={"John Smith": "Alex Johnson"},
                processing_time=2.5,
                total_pages=2,
            )

            assert report["document"] == "input.pdf"
            assert report["status"] == "success"
            assert report["total_pages"] == 2
            assert report["processing_time_seconds"] == 2.5
            assert len(report["masked_fields"]) == 3
            assert "statistics" in report

    def test_generate_report_with_warnings(self, generator, sample_fields):
        """Should include warnings in report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.pdf"
            input_path.touch()
            output_path.write_bytes(b"content")

            report = generator.generate_report(
                input_path=str(input_path),
                output_path=str(output_path),
                detected_fields=[],
                hash_mapping={},
                processing_time=1.0,
                total_pages=1,
                warnings=["Some warning", "Another warning"],
            )

            assert report["warnings"] == ["Some warning", "Another warning"]

    def test_generate_report_field_details(self, generator, sample_fields):
        """Should include all field details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.pdf"
            input_path.touch()
            output_path.write_bytes(b"content")

            report = generator.generate_report(
                input_path=str(input_path),
                output_path=str(output_path),
                detected_fields=sample_fields,
                hash_mapping={},
                processing_time=1.0,
                total_pages=2,
            )

            # Check first field
            field = report["masked_fields"][0]
            assert field["field_id"] == 1
            assert field["field_type"] == "person_name"
            assert field["page"] == 1
            assert field["confidence"] == 0.95
            assert field["detection_method"] == "llm_unified"
            assert field["regulations"] == ["GDPR", "KVKK"]
            assert field["risk_level"] == "HIGH"


class TestGetMaskingMethod:
    """Tests for masking method detection."""

    @pytest.fixture
    def generator(self):
        return ReportGenerator()

    def test_signature_uses_contour(self, generator):
        """Signature should use contour masking."""
        field = {"field_type": "signature"}
        assert generator._get_masking_method(field) == "contour"

    def test_stamp_uses_contour(self, generator):
        """Stamp should use contour masking."""
        field = {"field_type": "stamp"}
        assert generator._get_masking_method(field) == "contour"

    def test_visual_detection_uses_contour(self, generator):
        """Visual detection should use contour masking."""
        field = {"detection_method": "visual_detection"}
        assert generator._get_masking_method(field) == "contour"

    def test_text_uses_white_box(self, generator):
        """Text fields should use white_box_overlay."""
        field = {"field_type": "person_name"}
        assert generator._get_masking_method(field) == "white_box_overlay"

    def test_custom_masking_method(self, generator):
        """Should use field's masking_method if specified."""
        field = {"field_type": "email", "masking_method": "custom_method"}
        assert generator._get_masking_method(field) == "custom_method"


class TestCalculateStatistics:
    """Tests for statistics calculation."""

    @pytest.fixture
    def generator(self):
        return ReportGenerator()

    def test_empty_fields(self, generator):
        """Should handle empty fields."""
        stats = generator._calculate_statistics([])
        assert stats["total_masked_fields"] == 0
        assert stats["by_type"] == {}
        assert stats["by_page"] == {}

    def test_statistics_by_type(self, generator):
        """Should count by field type."""
        fields = [
            {"field_type": "person_name", "page": 1},
            {"field_type": "person_name", "page": 1},
            {"field_type": "email", "page": 1},
        ]
        stats = generator._calculate_statistics(fields)

        assert stats["total_masked_fields"] == 3
        assert stats["by_type"]["person_name"] == 2
        assert stats["by_type"]["email"] == 1

    def test_statistics_by_page(self, generator):
        """Should count by page."""
        fields = [
            {"field_type": "name", "page": 1},
            {"field_type": "name", "page": 1},
            {"field_type": "name", "page": 2},
        ]
        stats = generator._calculate_statistics(fields)

        assert stats["by_page"]["1"] == 2
        assert stats["by_page"]["2"] == 1

    def test_statistics_by_regulation(self, generator):
        """Should count by regulation."""
        fields = [
            {"field_type": "name", "page": 1, "regulations": ["GDPR", "KVKK"]},
            {"field_type": "name", "page": 1, "regulations": ["GDPR"]},
        ]
        stats = generator._calculate_statistics(fields)

        assert stats["by_regulation"]["GDPR"] == 2
        assert stats["by_regulation"]["KVKK"] == 1

    def test_statistics_by_risk_level(self, generator):
        """Should count by risk level."""
        fields = [
            {"field_type": "name", "page": 1, "risk_level": "HIGH"},
            {"field_type": "name", "page": 1, "risk_level": "HIGH"},
            {"field_type": "name", "page": 1, "risk_level": "MEDIUM"},
        ]
        stats = generator._calculate_statistics(fields)

        assert stats["by_risk_level"]["HIGH"] == 2
        assert stats["by_risk_level"]["MEDIUM"] == 1


class TestSaveReport:
    """Tests for report saving."""

    def test_save_report_basic(self):
        """Should save report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)
            report = {
                "document": "test.pdf",
                "status": "success",
                "statistics": {"total_masked_fields": 5},
            }

            output_path = generator.save_report(report)

            assert output_path.exists()
            assert output_path.name == "test_report.json"

            with open(output_path) as f:
                saved = json.load(f)
            assert saved["document"] == "test.pdf"

    def test_save_report_custom_name(self):
        """Should use custom output name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(output_dir=tmpdir)
            report = {"document": "test.pdf"}

            output_path = generator.save_report(report, output_name="custom.json")

            assert output_path.name == "custom.json"


class TestGenerateBatchSummary:
    """Tests for batch summary generation."""

    @pytest.fixture
    def generator(self):
        return ReportGenerator()

    def test_batch_summary_all_success(self, generator):
        """Should summarize successful batch."""
        reports = [
            {
                "status": "success",
                "total_pages": 5,
                "processing_time_seconds": 2.0,
                "statistics": {
                    "total_masked_fields": 10,
                    "by_type": {"person_name": 5, "email": 5},
                },
            },
            {
                "status": "success",
                "total_pages": 3,
                "processing_time_seconds": 1.5,
                "statistics": {
                    "total_masked_fields": 6,
                    "by_type": {"person_name": 3, "phone": 3},
                },
            },
        ]

        summary = generator.generate_batch_summary(reports)

        assert summary["total_documents"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["total_pages_processed"] == 8
        assert summary["total_fields_masked"] == 16
        assert summary["total_processing_time_seconds"] == 3.5
        assert summary["fields_by_type"]["person_name"] == 8

    def test_batch_summary_with_failures(self, generator):
        """Should include failed documents."""
        reports = [
            {"status": "success", "total_pages": 5, "processing_time_seconds": 2.0,
             "statistics": {"total_masked_fields": 10, "by_type": {}}},
            {"status": "failed", "document": "bad.pdf", "error": "OCR failed"},
        ]

        summary = generator.generate_batch_summary(reports)

        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert len(summary["failed_documents"]) == 1
        assert summary["failed_documents"][0]["document"] == "bad.pdf"

    def test_batch_summary_empty(self, generator):
        """Should handle empty batch."""
        summary = generator.generate_batch_summary([])

        assert summary["total_documents"] == 0
        assert summary["average_time_per_document"] == 0


class TestGenerateErrorReport:
    """Tests for error report generation."""

    @pytest.fixture
    def generator(self):
        return ReportGenerator()

    def test_error_report_basic(self, generator):
        """Should generate error report."""
        report = generator.generate_error_report(
            input_path="/path/to/file.pdf",
            error="File not found",
            processing_time=0.5,
        )

        assert report["document"] == "file.pdf"
        assert report["status"] == "failed"
        assert report["error"] == "File not found"
        assert report["processing_time_seconds"] == 0.5


class TestPrintSummary:
    """Tests for summary printing."""

    @pytest.fixture
    def generator(self):
        return ReportGenerator()

    def test_print_success_summary(self, generator, capsys):
        """Should print success summary."""
        report = {
            "document": "test.pdf",
            "status": "success",
            "total_pages": 5,
            "processing_time_seconds": 2.5,
            "output_path": "/output/test.pdf",
            "statistics": {
                "total_masked_fields": 10,
                "by_type": {"person_name": 5, "email": 5},
                "by_regulation": {"GDPR": 10},
            },
            "warnings": ["Some warning"],
        }

        generator.print_summary(report)
        captured = capsys.readouterr()

        assert "PROCESSING REPORT" in captured.out
        assert "test.pdf" in captured.out
        assert "success" in captured.out
        assert "5" in captured.out  # pages
        assert "person_name" in captured.out
        assert "Some warning" in captured.out

    def test_print_error_summary(self, generator, capsys):
        """Should print error summary."""
        report = {
            "document": "test.pdf",
            "status": "failed",
            "error": "Processing failed",
        }

        generator.print_summary(report)
        captured = capsys.readouterr()

        assert "failed" in captured.out
        assert "Processing failed" in captured.out
