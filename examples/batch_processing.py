#!/usr/bin/env python3
"""
Batch Processing Example

Demonstrates processing multiple PDF files in a folder.
"""

import asyncio
import sys
from pathlib import Path

from document_anonymizer import DocumentAnonymizer


async def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_processing.py <input_folder> <output_dir>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_dir = sys.argv[2]

    if not input_folder.is_dir():
        print(f"Not a directory: {input_folder}")
        sys.exit(1)

    # Find PDF files
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {input_folder}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files")

    # Initialize anonymizer
    anonymizer = DocumentAnonymizer()

    try:
        # Process all files
        reports = await anonymizer.anonymize_batch(
            str(input_folder),
            output_dir
        )

        # Summary
        success_count = sum(1 for r in reports if r.get("status") == "success")
        total_fields = sum(
            r.get("statistics", {}).get("total_masked_fields", 0)
            for r in reports
        )

        print(f"\nBatch complete: {success_count}/{len(reports)} successful")
        print(f"Total fields masked: {total_fields}")
        print(f"Output directory: {output_dir}")

    finally:
        await anonymizer.close()


if __name__ == "__main__":
    asyncio.run(main())
