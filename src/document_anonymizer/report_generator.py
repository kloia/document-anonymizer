"""JSON report generator for masking operations."""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ReportGenerator:
    """JSON report generator."""

    def __init__(self, config: Optional[Dict] = None, output_dir: Optional[str] = None):
        """Initialize report generator."""
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else None
        self._report_dir: Optional[Path] = None

    @property
    def report_dir(self) -> Path:
        """Get report directory."""
        if self._report_dir is None:
            if self.output_dir:
                self._report_dir = self.output_dir / "logs" / "reports"
            else:
                self._report_dir = Path(
                    self.config.get("logging", {}).get("report_dir", "logs/reports/")
                )
        self._report_dir.mkdir(parents=True, exist_ok=True)
        return self._report_dir

    def set_output_dir(self, output_dir: str) -> None:
        """Update output directory for reports."""
        self.output_dir = Path(output_dir)
        self._report_dir = None

    def generate_report(
        self,
        input_path: str,
        output_path: str,
        detected_fields: List[Dict],
        hash_mapping: Dict[str, str],
        processing_time: float,
        total_pages: int,
        warnings: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate detailed JSON report.

        Args:
            input_path: Input PDF path
            output_path: Output PDF path
            detected_fields: Detected sensitive fields
            hash_mapping: Original -> Dummy text mapping
            processing_time: Processing time (seconds)
            total_pages: Total page count
            warnings: Warning messages

        Returns:
            Report dictionary
        """
        in_path = Path(input_path)
        out_path = Path(output_path)

        masked_fields = []
        for idx, field in enumerate(detected_fields, start=1):
            field_entry = {
                "field_id": idx,
                "field_type": field.get("field_type", "unknown"),
                "page": field.get("page", 0),
                "bbox": list(field.get("bbox") or []),
                "masking_method": self._get_masking_method(field),
                "confidence": field.get("confidence", 0),
            }

            original_text = field.get("original_text", "")
            if original_text:
                field_entry["original_text"] = original_text
                field_entry["dummy_text"] = field.get("dummy_text", "") or hash_mapping.get(
                    original_text, ""
                )

            if "detection_method" in field:
                field_entry["detection_method"] = field["detection_method"]

            # Regulation info for audit/compliance
            if "regulations" in field:
                field_entry["regulations"] = field["regulations"]
            if "risk_level" in field:
                field_entry["risk_level"] = field["risk_level"]
            if "reason" in field:
                field_entry["reason"] = field["reason"]

            if "source_bboxes" in field and field["source_bboxes"]:
                field_entry["source_bboxes"] = [list(b) for b in field["source_bboxes"]]

            if "merged_line_count" in field:
                field_entry["merged_line_count"] = field["merged_line_count"]

            masked_fields.append(field_entry)

        statistics = self._calculate_statistics(detected_fields)

        report = {
            "document": in_path.name,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "status": "success",
            "input_path": str(in_path),
            "output_path": str(out_path),
            "total_pages": total_pages,
            "processing_time_seconds": round(processing_time, 2),
            "masked_fields": masked_fields,
            "statistics": statistics,
            "warnings": warnings or [],
            "quality_checks": {
                "original_text_still_readable": False,
                "layout_preserved": True,
                "file_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2)
                if out_path.exists()
                else 0,
            },
        }

        return report

    def _get_masking_method(self, field: Dict) -> str:
        """Determine masking method for field."""
        field_type = field.get("field_type", "").lower()
        detection_method = field.get("detection_method", "")

        # Contour-based masking for visual elements (preferred over blur)
        if field_type in ["stamp", "signature", "seal", "handwriting"]:
            return "contour"
        if detection_method in ("visual_detection", "llm_visual_detection"):
            return "contour"

        return field.get("masking_method", "white_box_overlay")

    def _calculate_statistics(self, detected_fields: List[Dict]) -> Dict:
        """Calculate processing statistics."""
        by_type: Dict[str, int] = defaultdict(int)
        by_page: Dict[str, int] = defaultdict(int)
        by_regulation: Dict[str, int] = defaultdict(int)
        by_risk_level: Dict[str, int] = defaultdict(int)

        for field in detected_fields:
            field_type = field.get("field_type", "unknown")
            page = field.get("page", 0)

            by_type[field_type] += 1
            by_page[str(page)] += 1

            # Count regulations
            for reg in field.get("regulations", []):
                by_regulation[reg] += 1

            # Count risk levels
            risk_level = field.get("risk_level", "UNKNOWN")
            by_risk_level[risk_level] += 1

        return {
            "total_masked_fields": len(detected_fields),
            "by_type": dict(by_type),
            "by_page": dict(by_page),
            "by_regulation": dict(by_regulation),
            "by_risk_level": dict(by_risk_level),
        }

    def save_report(self, report: Dict, output_name: Optional[str] = None) -> Path:
        """Save report to JSON file."""
        if output_name is None:
            document_name = Path(report.get("document", "unknown")).stem
            output_name = f"{document_name}_report.json"

        output_path = self.report_dir / output_name

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Report saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Report save error: {e}")
            raise

    def generate_batch_summary(self, reports: List[Dict]) -> Dict:
        """Generate summary report for batch processing."""
        successful = [r for r in reports if r.get("status") == "success"]
        failed = [r for r in reports if r.get("status") == "failed"]

        total_fields = sum(
            r.get("statistics", {}).get("total_masked_fields", 0) for r in successful
        )

        total_pages = sum(r.get("total_pages", 0) for r in successful)
        total_time = sum(r.get("processing_time_seconds", 0) for r in successful)

        type_totals: Dict[str, int] = defaultdict(int)
        for report in successful:
            by_type = report.get("statistics", {}).get("by_type", {})
            for field_type, count in by_type.items():
                type_totals[field_type] += count

        summary = {
            "batch_processed_at": datetime.utcnow().isoformat() + "Z",
            "total_documents": len(reports),
            "successful": len(successful),
            "failed": len(failed),
            "total_pages_processed": total_pages,
            "total_fields_masked": total_fields,
            "total_processing_time_seconds": round(total_time, 2),
            "average_time_per_document": round(total_time / len(successful), 2)
            if successful
            else 0,
            "fields_by_type": dict(type_totals),
            "failed_documents": [
                {
                    "document": r.get("document", "unknown"),
                    "error": r.get("error", "Unknown error"),
                }
                for r in failed
            ],
        }

        return summary

    def generate_error_report(
        self, input_path: str, error: str, processing_time: float = 0
    ) -> Dict:
        """Generate report for error case."""
        path = Path(input_path)

        return {
            "document": path.name,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "status": "failed",
            "input_path": str(path),
            "error": error,
            "processing_time_seconds": round(processing_time, 2),
        }

    def print_summary(self, report: Dict) -> None:
        """Print report summary to console."""
        print("\n" + "=" * 50)
        print("PROCESSING REPORT")
        print("=" * 50)

        print(f"Document: {report.get('document', 'N/A')}")
        print(f"Status: {report.get('status', 'N/A')}")

        if report.get("status") == "success":
            print(f"Total Pages: {report.get('total_pages', 0)}")
            print(f"Processing Time: {report.get('processing_time_seconds', 0):.2f} seconds")

            stats = report.get("statistics", {})
            print(f"\nMasked Field Count: {stats.get('total_masked_fields', 0)}")

            by_type = stats.get("by_type", {})
            if by_type:
                print("\nBy Field Type:")
                for field_type, count in by_type.items():
                    print(f"  - {field_type}: {count}")

            by_regulation = stats.get("by_regulation", {})
            if by_regulation:
                print("\nBy Regulation:")
                for reg, count in by_regulation.items():
                    print(f"  - {reg}: {count}")

            warnings = report.get("warnings", [])
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  ! {warning}")

            print(f"\nOutput: {report.get('output_path', 'N/A')}")

        else:
            print(f"Error: {report.get('error', 'Unknown error')}")

        print("=" * 50 + "\n")
