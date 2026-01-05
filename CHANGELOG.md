# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source community files (LICENSE, SECURITY.md, CODE_OF_CONDUCT.md)
- GitHub issue and PR templates
- CI/CD workflows for testing and releases
- Comprehensive test suite
- Pre-commit hooks configuration

## [0.1.0] - 2024-12-30

### Added
- Initial release of Document Anonymizer
- Vision LLM-powered sensitive data detection (Qwen VL)
- EasyOCR integration for text extraction
- Format-preserving dummy data generation
- Support for any country's documents without configuration
- Contour-based masking for signatures and stamps
- Two-stage detection with confidence filtering
- Manual review queue for medium-confidence detections
- Token registry for cross-document consistency
- HMAC-SHA256 deterministic tokenization
- Post-masking verification
- Batch processing with concurrency control
- CLI tool (`docanon`) with multiple modes
- Compliance support: GDPR, KVKK, CCPA, LGPD, PDPA
- Detailed JSON reports with statistics
- Async-first architecture

### Security
- No original sensitive values stored (only tokens)
- Configurable secret key for tokenization
- Input validation throughout the pipeline

[Unreleased]: https://github.com/kloia/document-anonymizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kloia/document-anonymizer/releases/tag/v0.1.0
