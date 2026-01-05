# LLM Prompts

This directory contains the prompt template used by the LLM-based detection system.

## sensitive_field_detector.md

Prompt for detecting all sensitive elements in a document:
- **Text-based PII**: Names, addresses, phone numbers, emails, IDs, etc.
- **Visual elements**: Signatures, stamps, seals

The prompt receives OCR-extracted text blocks and the document image, returning structured JSON with detected sensitive fields.

## Why Externalized

Prompts are stored as separate files to ensure:
- Transparency and auditability
- Easy iteration without code changes
- Clear separation between logic and instruction

## Limitations

- LLM outputs are probabilistic
- This tool does NOT guarantee full legal compliance
- Human review is recommended for critical documents
