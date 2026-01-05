"""
Document Anonymizer

A general-purpose document anonymization system for scanned and digital documents.
Uses visual-based masking with OCR and LLM-powered sensitive data detection.

Features:
- Async-first architecture for high performance
- HMAC-SHA256 based deterministic tokenization
- Cross-document consistency
- Integration with ocr-preprocessor for enhanced image processing
- Pattern-based and LLM-based sensitive field detection
- Post-masking verification
"""

__version__ = "0.1.0"
__author__ = "Document Anonymizer Team"

from .anonymization_engine import AnonymizationEngine, create_anonymization_engine
from .constants import DocumentType, EntityNamespace
from .document_anonymizer import DocumentAnonymizer
from .dummy_generator import DummyDataGenerator
from .field_detector import FieldDetector
from .image_masker import ImageMasker
from .ocr_processor import OCRProcessor
from .pdf_handler import get_pdf_info, images_to_pdf, pdf_to_images
from .report_generator import ReportGenerator
from .text_renderer import TextRenderer
from .verification import PostMaskingVerifier, VerificationResult, VerificationStatus

__all__ = [
    # Version
    "__version__",
    # Main class
    "DocumentAnonymizer",
    # Core components
    "AnonymizationEngine",
    "create_anonymization_engine",
    "DummyDataGenerator",
    "OCRProcessor",
    "FieldDetector",
    "ImageMasker",
    "TextRenderer",
    # PDF handling
    "pdf_to_images",
    "images_to_pdf",
    "get_pdf_info",
    # Reporting and verification
    "ReportGenerator",
    "PostMaskingVerifier",
    "VerificationStatus",
    "VerificationResult",
    # Constants and enums
    "EntityNamespace",
    "DocumentType",
]
