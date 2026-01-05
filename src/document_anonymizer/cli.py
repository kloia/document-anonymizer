"""
Command-line interface for document-anonymizer.

Usage:
    docanon input.pdf -o output.pdf
    docanon ./documents/ -o ./anonymized/
    docanon invoice.pdf --dry-run
    docanon batch/ -o results/ --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .document_anonymizer import DocumentAnonymizer
from .report_generator import ReportGenerator


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity settings."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="docanon",
        description="Anonymize sensitive data in scanned and digital documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  docanon invoice.pdf -o invoice_anon.pdf
  docanon invoice.pdf -o ./output/
  docanon ./documents/ -o ./anonymized/
  docanon scan.pdf --dry-run
  docanon batch/ -o results/ --verbose
  docanon doc.pdf --config custom_rules.yaml
        """,
    )

    parser.add_argument(
        "input",
        help="Input PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file or directory"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Custom configuration YAML file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only, do not mask (shows what would be anonymized)"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip interactive review for medium confidence fields"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (debug logging)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (only show errors)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-masking verification"
    )
    parser.add_argument(
        "--save-registry",
        default=None,
        help="Path to save token registry (for cross-document consistency)"
    )
    parser.add_argument(
        "--load-registry",
        default=None,
        help="Path to load existing token registry"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    return parser


def print_banner() -> None:
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              DOCUMENT ANONYMIZATION SYSTEM                   ║
║            Visual Masking for Scanned Documents              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


async def interactive_review(fields: list) -> list:
    """
    Interactive CLI review for medium confidence fields.

    Args:
        fields: List of fields needing review

    Returns:
        List of approved fields to mask
    """
    if not fields:
        return []

    print("\n" + "=" * 60)
    print("  MANUAL REVIEW REQUIRED")
    print("  The following fields have medium confidence (0.60-0.85)")
    print("=" * 60)

    approved = []

    for i, field in enumerate(fields, 1):
        print(f"\n[{i}/{len(fields)}] Field Review:")
        print(f"  Text: {field.get('text', 'N/A')}")
        print(f"  Type: {field.get('field_type', 'unknown')}")
        print(f"  Confidence: {field.get('confidence', 0):.2f}")
        print(f"  Page: {field.get('page', 'N/A')}")
        if field.get('reason'):
            print(f"  Reason: {field.get('reason')}")

        while True:
            response = input("\n  Mask this field? [Y/n/a/s/?]: ").strip().lower()

            if response in ('', 'y', 'yes'):
                approved.append(field)
                print("  → Approved for masking")
                break
            elif response in ('n', 'no'):
                print("  → Skipped")
                break
            elif response in ('a', 'all'):
                # Approve all remaining
                approved.append(field)
                approved.extend(fields[i:])
                print(f"  → Approved all remaining ({len(fields) - i + 1} fields)")
                return approved
            elif response in ('s', 'skip'):
                # Skip all remaining
                print(f"  → Skipped all remaining ({len(fields) - i + 1} fields)")
                return approved
            elif response == '?':
                print("\n  Commands:")
                print("    Y/Enter - Yes, mask this field")
                print("    N       - No, skip this field")
                print("    A       - Approve ALL remaining fields")
                print("    S       - Skip ALL remaining fields")
                print("    ?       - Show this help")
            else:
                print("  Invalid input. Type ? for help.")

    print(f"\n✓ Review complete: {len(approved)}/{len(fields)} fields approved")
    return approved


async def run_anonymization(args: argparse.Namespace) -> int:
    """Run the anonymization process."""
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    # Validate input
    if not input_path.exists():
        logging.error(f"Input not found: {input_path}")
        return 1

    if input_path.is_file() and not input_path.suffix.lower() == '.pdf':
        logging.error(f"Input must be a PDF file: {input_path}")
        return 1

    # Determine output directory
    if input_path.is_file():
        if output_path.suffix.lower() == '.pdf':
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find config file
    config_path = args.config
    if config_path is None:
        package_config = Path(__file__).parent / "config" / "masking_rules.yaml"
        local_config = Path("config/masking_rules.yaml")

        if package_config.exists():
            config_path = str(package_config)
        elif local_config.exists():
            config_path = str(local_config)

    # Initialize anonymizer
    try:
        anonymizer = DocumentAnonymizer(config_path)
    except Exception as e:
        logging.error(f"Failed to initialize anonymizer: {e}")
        return 1

    # Load registry if specified
    if args.load_registry:
        try:
            anonymizer.load_token_registry(args.load_registry)
            logging.info(f"Token registry loaded: {args.load_registry}")
        except Exception as e:
            logging.warning(f"Failed to load registry: {e}")

    # Start timing
    start_time = time.time()

    # Process
    if input_path.is_file():
        # Single file
        logging.info(f"Processing: {input_path.name}")

        if args.dry_run:
            report = await anonymizer.analyze_document(str(input_path))
            print("\n[DRY RUN] Analysis complete (no masking performed)")
        else:
            # Determine review callback
            review_callback = None
            if not args.no_review:
                review_callback = interactive_review

            report = await anonymizer.anonymize_document(
                str(input_path),
                str(output_dir),
                review_callback=review_callback
            )

        # Calculate duration
        duration = time.time() - start_time

        # Print report
        report_gen = ReportGenerator(anonymizer.config)
        report_gen.print_summary(report)

        # Save registry if requested
        if args.save_registry:
            anonymizer.save_token_registry(args.save_registry)
            logging.info(f"Token registry saved: {args.save_registry}")

        if report.get('status') == 'success':
            print(f"\n✓ Output: {report.get('output_path')}")
            print(f"✓ Completed in {format_duration(duration)}")
            return 0
        elif report.get('status') == 'dry_run':
            auto_count = report.get('auto_mask_count', 0)
            review_count = report.get('needs_review_count', 0)
            total = len(report.get('detected_fields', []))
            print(f"\n✓ Detected {total} sensitive fields:")
            print(f"  - Auto-mask (high confidence): {auto_count}")
            print(f"  - Needs review (medium confidence): {review_count}")
            print(f"✓ Analysis completed in {format_duration(duration)}")
            return 0
        else:
            logging.error(f"Processing failed: {report.get('error')}")
            return 1

    else:
        # Batch processing
        logging.info(f"Processing directory: {input_path}")

        reports = await anonymizer.anonymize_batch(
            str(input_path),
            str(output_dir)
        )

        # Calculate duration
        duration = time.time() - start_time

        # Summary
        successful = sum(1 for r in reports if r.get('status') == 'success')
        failed = sum(1 for r in reports if r.get('status') == 'failed')

        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total files:  {len(reports)}")
        print(f"Successful:   {successful}")
        print(f"Failed:       {failed}")
        print(f"Duration:     {format_duration(duration)}")

        if failed > 0:
            print("\nFailed files:")
            for r in reports:
                if r.get('status') == 'failed':
                    print(f"  ✗ {r.get('document')}: {r.get('error')}")

        # Save registry
        if args.save_registry:
            anonymizer.save_token_registry(args.save_registry)
            logging.info(f"Token registry saved: {args.save_registry}")

        print("=" * 60)
        print(f"\n✓ Output directory: {output_dir}")
        print(f"✓ Completed in {format_duration(duration)}")

        return 0 if failed == 0 else 1


def main() -> int:
    """Main CLI entry point."""
    # Load environment variables
    load_dotenv()

    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Print banner (unless quiet)
    if not args.quiet:
        print_banner()

    # Run async processing
    try:
        return asyncio.run(run_anonymization(args))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
