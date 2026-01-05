# Architecture Overview

## Processing Pipeline

```
PDF Input
    │
    ├──► PDF to Images (PyMuPDF @ 300 DPI)
    │
    ├──► Language Detection (LLM-based)
    │         │
    │         └──► Updates OCR languages and Faker locale
    │
    ├──► OCR Processing (EasyOCR + ocr-preprocessor)
    │         │
    │         └──► Text blocks with bounding boxes
    │
    ├──► Sensitive Field Detection
    │         ├──► LLM Detection (if API key configured)
    │         └──► Pattern Matching (fallback)
    │
    ├──► Confidence Filtering
    │         ├──► ≥0.85: Auto-mask
    │         ├──► 0.60-0.85: Manual review
    │         └──► <0.60: Ignore
    │
    ├──► Masking & Dummy Data Rendering
    │         ├──► Contour-based masking (signatures/stamps)
    │         ├──► White box overlay (text fields)
    │         └──► Format-preserving dummy text
    │
    ├──► Post-Masking Verification (OCR)
    │
    └──► PDF Output
```

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `document_anonymizer.py` | Main orchestrator, coordinates all components |
| `pdf_handler.py` | PDF to image conversion, image to PDF output |
| `ocr_processor.py` | EasyOCR integration with preprocessing |
| `field_detector.py` | Sensitive field detection (LLM + patterns) |
| `llm_classifier.py` | Vision LLM API client |
| `anonymization_engine.py` | HMAC tokenization and consistency |
| `dummy_generator.py` | Format-preserving dummy data |
| `text_renderer.py` | Font matching and text rendering |
| `image_masker.py` | Visual masking methods |
| `verification.py` | Post-masking leak detection |
| `constants.py` | Entity types, patterns, mappings |

## Data Flow

1. **Input**: PDF file path
2. **Conversion**: PDF pages → RGB images
3. **Detection**: Images → sensitive field coordinates
4. **Masking**: Original images + fields → masked images
5. **Output**: Masked images → PDF file

## Key Design Decisions

- **Async-first**: All I/O operations are async (httpx, aiofiles)
- **Deterministic**: HMAC-based tokenization ensures same input → same output
- **Cross-document consistency**: Token registry maintains entity mappings
- **Format-preserving**: Dummy data matches original structure
- **Non-reversible masking**: Contour-based approach prevents recovery
