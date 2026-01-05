#!/usr/bin/env python3
"""
Basic Usage Example

Demonstrates single file anonymization with document-anonymizer.
"""

import asyncio
import sys
from pathlib import Path

from document_anonymizer import DocumentAnonymizer


async def main():
    if len(sys.argv) < 2:
        print("Usage: python basic_usage.py <pdf_file> [output_dir]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"

    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    # Initialize anonymizer
    # LLM settings are read from environment variables:
    # - LLM_API_KEY
    # - LLM_API_URL
    # - LLM_MODEL_VISION
    anonymizer = DocumentAnonymizer()

    try:
        # Process document
        print(f"Processing: {input_file}")
        report = await anonymizer.anonymize_document(
            str(input_file),
            output_dir
        )

        # Print results
        if report.get("status") == "success":
            stats = report.get("statistics", {})
            print(f"Success! Masked {stats.get('total_masked_fields', 0)} fields")
            print(f"Output: {report.get('output_path')}")
        else:
            print(f"Error: {report.get('error')}")

    finally:
        await anonymizer.close()


if __name__ == "__main__":
    asyncio.run(main())
