# Examples

This directory contains example scripts demonstrating how to use the document-anonymizer library.

## Files

- `basic_usage.py` - Simple single file anonymization
- `batch_processing.py` - Process multiple files in a folder

## Setup

Before running examples, ensure you have installed the package:

```bash
pip install -e ..
```

Optionally, configure LLM for enhanced detection:

```bash
export LLM_API_KEY=your_api_key
export LLM_API_URL=https://api.openai.com/v1
```

## Running Examples

```bash
# Basic usage
python basic_usage.py path/to/document.pdf

# Batch processing
python batch_processing.py path/to/folder/ ./output/
```

## Sample Documents

Place test PDF files in `sample_documents/` directory. These files are gitignored to prevent accidental commits of sensitive data.
