# Document Anonymizer

[![CI](https://github.com/kloia/document-anonymizer/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/kloia/document-anonymizer/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kloia/document-anonymizer/branch/master/graph/badge.svg)](https://codecov.io/gh/kloia/document-anonymizer)
[![PyPI version](https://badge.fury.io/py/document-anonymizer.svg)](https://badge.fury.io/py/document-anonymizer)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A general-purpose document anonymization framework for scanned and digital documents. Detects sensitive data via pattern matching and optional Vision LLM, then applies non-reversible masking with format-preserving dummy data generation.

## What This Project Does

- Extracts text and bounding boxes from PDF documents using OCR (EasyOCR)
- Detects sensitive fields via **pattern matching** (works standalone) or **Vision LLM** (optional enhancement)
- Masks detected regions with contour-based or overlay methods (non-reversible)
- Generates format-preserving dummy data to replace masked content
- Verifies output via OCR to detect potential data leakage

## What This Project Does NOT Do

- Domain-specific processing (not trade-finance, banking, or industry-specific)
- Intelligent table cell detection (tables processed as text blocks)
- Automatic multi-line address clustering (each OCR block processed independently)
- Guarantee 100% detection accuracy (pattern-based detection has known limitations)

## Architecture

```
PDF Input
    │
    ▼
┌─────────────────┐
│ PDF to Images   │  PyMuPDF @ 300 DPI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ OCR Processing  │  EasyOCR + optional preprocessing
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│           DETECTION LAYER                   │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │ Pattern Matching│  │ Vision LLM       │  │
│  │ (always active) │  │ (if API key set) │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           └──────────┬─────────┘            │
└──────────────────────┼──────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ Confidence Filtering    │
         │ ≥0.85: Auto-mask        │
         │ 0.60-0.85: Manual review│
         │ <0.60: Ignore           │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ Masking & Rendering     │
         │ • Contour-based (visual)│
         │ • White box overlay     │
         │ • Format-preserving     │
         │   dummy text            │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ Post-Masking Verify     │
         │ OCR output to detect    │
         │ potential leakage       │
         └────────────┬────────────┘
                      │
                      ▼
                 PDF Output
```

## Detection Capabilities

### Pattern-Based Detection (No LLM Required)

Works out-of-the-box with regex patterns for:

| Category | Supported Countries | Examples |
|----------|---------------------|----------|
| **Email** | Global | `user@example.com` |
| **Phone** | US, TR, DE, FR, UK, RU | `+1 555 123 4567`, `+90 532 123 45 67` |
| **National ID** | US (SSN), UK (NINO), FR (INSEE), IT (CF), ES (DNI) | `123-45-6789`, `AB123456C` |
| **License Plate** | TR, UK, FR, IT, ES, RU | `34 ABC 123`, `AB12 CDE`, `AA-123-AA` |
| **Postal Code** | UK | `SW1A 1AA` |
| **IP Address** | Global | `192.168.1.1` |
| **Company Names** | Global | Via 60+ legal suffixes (Ltd, GmbH, A.S., Inc...) |

### Vision LLM Detection (Optional)

When `LLM_API_KEY` is configured:
- Semantic understanding of context
- Signature and stamp detection
- Handwritten text recognition
- Document type classification

**Without LLM:** Pattern matching provides baseline detection. Signatures/stamps will not be detected but other text fields are masked if they match patterns.

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Quick Start

```bash
git clone https://github.com/kloia/document-anonymizer.git
cd document-anonymizer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install
pip install -e .

# Copy and configure environment
cp .env.example .env
```

### Verify Installation

```bash
docanon --version
```

## Usage

### Basic Usage (Pattern Detection Only)

```bash
# Single file - no LLM required
docanon document.pdf -o ./output/

# Batch processing
docanon ./documents/ -o ./anonymized/

# Dry run - analyze without masking
docanon document.pdf -o ./output/ --dry-run

# Skip manual review
docanon document.pdf -o output/ --no-review
```

### With Vision LLM

Set environment variables in `.env`:

```bash
LLM_API_URL=https://your-llm-endpoint.com
LLM_API_KEY=your_api_key
LLM_MODEL_VISION=Qwen/Qwen2.5-VL-72B-Instruct
```

Then run normally - LLM detection activates automatically.

### Manual Review Mode

Fields with 0.60-0.85 confidence prompt for review:

```
[3/5] Field Review:
  Text: John Smith
  Type: PERSON_NAME
  Confidence: 0.72

  Mask this field? [Y/n/a/s/?]:
```

- `Y` - Mask this field
- `N` - Skip this field
- `A` - Approve all remaining
- `S` - Skip all remaining

### Python API

```python
import asyncio
from document_anonymizer import DocumentAnonymizer

async def main():
    # Option 1: Using environment variables (recommended for production)
    # Set LLM_API_KEY and LLM_API_URL in your environment
    anonymizer = DocumentAnonymizer()

    # Option 2: Using explicit parameters (useful for development/testing)
    anonymizer = DocumentAnonymizer(
        llm_api_key="sk-your-api-key",
        llm_api_url="https://api.openai.com/v1",
        llm_model="gpt-4o",
        secret_key="your-secret-key"
    )

    # Process document
    report = await anonymizer.anonymize_document(
        "document.pdf",
        "output_dir"
    )

    print(f"Masked fields: {report['statistics']['total_masked_fields']}")

asyncio.run(main())
```

**Note:** LLM is optional. Without LLM configuration, pattern-based detection still works for emails, phones, IDs, etc.

## Configuration

Configuration priority (highest to lowest):
1. Constructor parameters
2. Environment variables
3. Config file
4. Default values

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | No | - | Vision LLM API key |
| `LLM_API_URL` | No | - | Vision LLM endpoint URL |
| `LLM_MODEL_VISION` | No | gpt-4o | Model identifier |
| `ANONYMIZATION_SECRET_KEY` | No* | (default key) | HMAC key for tokenization |
| `MAX_CONCURRENT_PAGES` | No | 8 | Page processing concurrency |

*Change in production for security.

### Configuration File

Create `masking_rules.yaml` for advanced options:

```yaml
detection_rules:
  use_llm_classification: true   # Enable if API key available
  use_fallback_detection: true   # Always use pattern matching
  min_confidence: 0.60           # Threshold for manual review
  auto_mask_confidence: 0.85     # Auto-mask above this

masking_strategy:
  text_fields:
    method: 'white_box_overlay'  # or 'background_aware'
  visual_fields:
    stamp_method: 'contour'      # Non-reversible masking

verification:
  enabled: true
  check_original_text: true      # OCR output to verify
```

## Format-Preserving Dummy Data

Replacements match the original format structure:

| Type | Original | Replacement |
|------|----------|-------------|
| License Plate | `34 KLY 482` | `50 ZVP 897` |
| Phone | `+90 532 123 4567` | `+41 405 854 8371` |
| Email | `john@example.com` | `alex@sample.org` |
| Company | `Acme Ltd` | `Delta Ltd` |

Same input always produces the same replacement (HMAC-based determinism).

## Output Structure

```
output_dir/
├── document_anonymized.pdf    # Masked document
├── token_registry.json        # Token mapping (no originals stored)
└── logs/
    └── reports/
        └── document_report.json
```

### Token Registry

Privacy-compliant format - original values are NOT stored:

```json
{
  "tokens": {
    "a1b2c3d4...": {
      "token": "Alex Johnson",
      "namespace": "PERSON_NAME",
      "normalized_hash": "sha256...",
      "confidence": 0.96
    }
  }
}
```

## Limitations & Disclaimers

### Detection Limitations

1. **Visual elements require LLM** - Signatures and stamps are only detected when Vision LLM is configured
2. **Address lines processed independently** - Multi-line addresses are detected as separate blocks
3. **Tables processed as text** - No cell-level detection; tabular data treated as regular text
4. **OCR accuracy varies** - Low-quality scans may produce incomplete text extraction

### Security Disclaimer

- **No 100% detection guarantee** - This tool may miss some sensitive data. Always review outputs for critical use cases.
- **Change default secret key** - The default `ANONYMIZATION_SECRET_KEY` is for development only. Use a strong, unique key in production.
- **Not a compliance solution** - This tool assists with anonymization but does not guarantee GDPR/KVKK/CCPA compliance.
- **Token registry security** - The token registry does not store original values, but protect it as it contains replacement patterns.

## Compliance Notes

The system supports fields relevant to:
- GDPR (EU)
- KVKK (Turkey)
- CCPA (California)
- LGPD (Brazil)

Each masked field includes applicable regulation tags in reports.

**Note:** This tool assists with anonymization but does not guarantee compliance. Human review of outputs is recommended for sensitive use cases.

## Troubleshooting

### EasyOCR GPU Issues

```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

### First Run Slow

EasyOCR downloads language models on first use (~1GB). This is normal.

### Pattern Detection Only

If LLM API is not available, pattern matching activates automatically. No configuration change needed.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linter
ruff check src/

# Type checking
mypy src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

Report vulnerabilities via [SECURITY.md](SECURITY.md) - do not open public issues.
