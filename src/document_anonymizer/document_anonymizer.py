"""
Document Anonymizer - Main Orchestrator

Async-first document anonymization system for sensitive data masking.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .anonymization_engine import create_anonymization_engine
from .field_detector import FieldDetector
from .image_masker import ImageMasker
from .ocr_processor import OCRProcessor
from .pdf_handler import get_pdf_info, images_to_pdf, pdf_to_images
from .report_generator import ReportGenerator
from .text_renderer import TextRenderer
from .verification import PostMaskingVerifier, VerificationStatus

logger = logging.getLogger(__name__)


class DocumentAnonymizer:
    """Async document anonymizer with visual masking and deterministic tokenization."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        llm_api_key: Optional[str] = None,
        llm_api_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """Initialize anonymizer. Explicit params override environment variables."""
        self.config = self._load_config(config_path)

        # Store LLM settings (explicit params override env vars)
        self.config['llm'] = {
            'api_key': llm_api_key or os.getenv('LLM_API_KEY', ''),
            'api_url': llm_api_url or os.getenv('LLM_API_URL', ''),
            'model': llm_model or os.getenv('LLM_MODEL_VISION', 'gpt-4-vision-preview'),
        }

        # Store secret key (explicit param overrides env var)
        if secret_key:
            self.config['anonymization']['secret_key'] = secret_key

        # Create anonymization engine first (shared across components)
        registry_path = self.config.get('anonymization', {}).get('registry_path')
        anon_secret_key = self.config.get('anonymization', {}).get('secret_key')
        self._anon_engine = create_anonymization_engine(
            secret_key=anon_secret_key,
            registry_path=registry_path
        )

        # Initialize components
        self.ocr_processor = OCRProcessor(self.config)
        self.field_detector = FieldDetector(self.config)
        self.image_masker = ImageMasker(self.config)
        self.text_renderer = TextRenderer(
            self.config,
            anonymization_engine=self._anon_engine,
            image_masker=self.image_masker
        )
        self.verifier = PostMaskingVerifier(self.config)
        self.report_generator = ReportGenerator(self.config)

        # Concurrency settings
        self.max_concurrent_pages = int(os.getenv('MAX_CONCURRENT_PAGES', '8'))
        self.max_concurrent_files = int(os.getenv('MAX_CONCURRENT_FILES', '4'))
        self.streaming_threshold = int(os.getenv('STREAMING_THRESHOLD_PAGES', '20'))

        # State
        self._output_dir: Optional[Path] = None

        # Log initialization status
        llm_configured = bool(self.config['llm']['api_key'])
        logger.info(f"DocumentAnonymizer initialized (LLM: {'enabled' if llm_configured else 'disabled'})")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        default_config = {
            'anonymization': {
                'secret_key': None,
                'registry_path': None,  # Set dynamically in _configure_output_paths
                'persist_registry': True,
            },
            'ocr_settings': {
                'dpi': 300,
                'preprocessing': {
                    'enabled': True,
                    'use_ocr_preprocessor': True,
                    'pipeline': 'full',
                },
            },
            'detection_rules': {
                'use_llm_classification': True,
                'use_fallback_detection': True,
                'min_confidence': 0.5,
            },
            'masking_strategy': {
                'text_fields': {
                    'background_color': [255, 255, 255],
                    'text_color': [0, 0, 0],
                    'padding': 5,
                },
            },
            'verification': {
                'enabled': True,
                'strict_mode': False,
                'check_original_text': True,
            },
            'processing': {
                'max_concurrent_pages': 8,
                'max_concurrent_files': 4,
                'streaming_threshold_pages': 20,
            },
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        self._deep_merge(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        return default_config

    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge update into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _configure_output_paths(self, output_dir: str) -> None:
        """Configure output directory structure."""
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Logs directory (inside output)
        logs_dir = self._output_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)

        # Reports directory
        reports_dir = logs_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)

        # Update report generator
        self.report_generator.set_output_dir(str(self._output_dir))

        # Token registry path (inside output, not logs)
        registry_path = self._output_dir / 'token_registry.json'
        self.config['anonymization']['registry_path'] = str(registry_path)

        # Update engine registry path
        self._anon_engine.registry_path = str(registry_path)

    async def anonymize_document(
        self,
        file_path: str,
        output_dir: str,
        dry_run: bool = False,
        review_callback=None,
        auto_approve_unreviewed: bool = True
    ) -> Dict:
        """
        Anonymize a single document.

        Args:
            file_path: Input PDF file path
            output_dir: Output directory
            dry_run: If True, analyze only without masking
            review_callback: Optional callback for manual review of medium confidence fields.
                           Signature: async def callback(fields: List[Dict]) -> List[Dict]
                           Returns approved fields to mask.
            auto_approve_unreviewed: If True and no review_callback provided, automatically
                           approve all medium confidence fields. Default True for safety
                           (better to mask more than miss sensitive data).

        Returns:
            Processing report dictionary
        """
        start_time = time.time()
        file_path = Path(file_path)

        if not file_path.exists():
            return self.report_generator.generate_error_report(
                str(file_path), f"File not found: {file_path}"
            )

        self._configure_output_paths(output_dir)

        try:
            logger.info(f"Processing: {file_path.name}")

            # Get PDF info
            pdf_info = get_pdf_info(str(file_path))
            total_pages = pdf_info['page_count']

            # Convert PDF to images
            images = pdf_to_images(str(file_path))

            if not images:
                return self.report_generator.generate_error_report(
                    str(file_path), "Failed to convert PDF to images"
                )

            # Process pages - Phase 1: Detection
            all_auto_mask = []
            all_needs_review = []
            page_ocr_data = []
            hash_mapping = {}
            warnings = []

            # Detect document language using LLM and update processors
            if images:
                lang_info = await self.field_detector.detect_document_language(images[0])
                self.ocr_processor.update_languages(lang_info['languages'])
                self._anon_engine.update_locale(lang_info['locale'])

            for page_num, image in enumerate(images, start=1):
                logger.debug(f"Processing page {page_num}/{total_pages} - Detection phase")

                # Run OCR
                ocr_result = await self.ocr_processor.run_ocr(image, page_num)
                text_blocks = ocr_result.get('text_blocks', [])

                # Detect sensitive fields (returns tuple now)
                auto_mask, needs_review = await self.field_detector.detect_sensitive_fields(
                    image, text_blocks, page_num
                )

                all_auto_mask.extend(auto_mask)
                all_needs_review.extend(needs_review)
                page_ocr_data.append({
                    'page_num': page_num,
                    'image': image,
                    'text_blocks': text_blocks,
                })

            if dry_run:
                processing_time = time.time() - start_time
                return {
                    'document': file_path.name,
                    'status': 'dry_run',
                    'total_pages': total_pages,
                    'detected_fields': all_auto_mask + all_needs_review,
                    'auto_mask_count': len(all_auto_mask),
                    'needs_review_count': len(all_needs_review),
                    'processing_time_seconds': round(processing_time, 2),
                }

            # Manual review callback for medium confidence fields
            approved_review_fields = []
            if all_needs_review:
                if review_callback:
                    # Use provided callback for manual review
                    try:
                        approved_review_fields = await review_callback(all_needs_review)
                        logger.info(f"Manual review: {len(approved_review_fields)}/{len(all_needs_review)} approved")
                    except Exception as e:
                        logger.warning(f"Review callback error: {e}")
                        warnings.append(f"Manual review error: {str(e)}")
                elif auto_approve_unreviewed:
                    # Auto-approve all medium confidence fields (safer default)
                    approved_review_fields = all_needs_review
                    logger.info(f"Auto-approved {len(approved_review_fields)} medium confidence fields")

            # Combine fields to mask
            fields_to_mask = all_auto_mask + approved_review_fields

            # Phase 2: Masking
            all_fields = []
            masked_images = []

            for page_data in page_ocr_data:
                page_num = page_data['page_num']
                image = page_data['image']

                logger.debug(f"Processing page {page_num}/{total_pages} - Masking phase")

                # Get fields for this page
                page_fields = [f for f in fields_to_mask if f.get('page') == page_num]

                # Mask and render
                masked_image = image.copy()

                for field in page_fields:
                    try:
                        masked_image, dummy_text = self.text_renderer.mask_and_render(
                            masked_image, field,
                            hash_mapping=hash_mapping,
                            document_id=file_path.name
                        )

                        field['dummy_text'] = dummy_text
                        all_fields.append(field)

                    except Exception as e:
                        logger.warning(f"Field masking error: {e}")
                        warnings.append(f"Page {page_num}: Field masking error - {str(e)}")

                masked_images.append(masked_image)

            # Post-masking verification
            verification_result = await self.verifier.verify_masked_document(
                masked_images, all_fields
            )

            if verification_result.status == VerificationStatus.FAILED:
                warnings.append("Verification failed: possible data leakage detected")

            # Save masked PDF
            output_filename = f"{file_path.stem}_anonymized.pdf"
            output_path = self._output_dir / output_filename

            images_to_pdf(masked_images, str(output_path))

            # Save token registry
            if self.config['anonymization'].get('persist_registry', True):
                self._anon_engine.save_registry()

            # Generate report
            processing_time = time.time() - start_time

            report = self.report_generator.generate_report(
                input_path=str(file_path),
                output_path=str(output_path),
                detected_fields=all_fields,
                hash_mapping=hash_mapping,
                processing_time=processing_time,
                total_pages=total_pages,
                warnings=warnings,
            )

            # Add verification info
            report['verification'] = {
                'status': verification_result.status.value,
                'confidence_score': verification_result.confidence_score,
            }

            # Save report
            self.report_generator.save_report(report, f"{file_path.stem}_report.json")

            logger.info(f"Completed: {file_path.name} ({len(all_fields)} fields masked)")

            return report

        except Exception as e:
            logger.exception(f"Document processing error: {e}")
            processing_time = time.time() - start_time
            return self.report_generator.generate_error_report(
                str(file_path), str(e), processing_time
            )

    async def analyze_document(self, file_path: str) -> Dict:
        """
        Analyze document without masking (dry run).

        Args:
            file_path: Input PDF file path

        Returns:
            Analysis report
        """
        return await self.anonymize_document(file_path, "analysis_output", dry_run=True)

    async def anonymize_batch(
        self,
        folder_path: str,
        output_dir: str,
        review_callback=None,
        auto_approve_unreviewed: bool = True
    ) -> List[Dict]:
        """
        Anonymize multiple documents in a folder.

        Args:
            folder_path: Input folder path
            output_dir: Output directory
            review_callback: Optional callback for manual review
            auto_approve_unreviewed: Auto-approve medium confidence fields if no callback

        Returns:
            List of processing reports
        """
        folder_path = Path(folder_path)

        if not folder_path.is_dir():
            logger.error(f"Not a directory: {folder_path}")
            return []

        # Find PDF files
        pdf_files = list(folder_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in: {folder_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files")

        # Process files with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_files)

        async def process_with_semaphore(pdf_file: Path) -> Dict:
            async with semaphore:
                return await self.anonymize_document(
                    str(pdf_file),
                    output_dir,
                    review_callback=review_callback,
                    auto_approve_unreviewed=auto_approve_unreviewed
                )

        tasks = [process_with_semaphore(f) for f in pdf_files]
        reports = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_reports = []
        for i, result in enumerate(reports):
            if isinstance(result, Exception):
                processed_reports.append(
                    self.report_generator.generate_error_report(
                        str(pdf_files[i]), str(result)
                    )
                )
            else:
                processed_reports.append(result)

        # Generate batch summary
        summary = self.report_generator.generate_batch_summary(processed_reports)
        summary_path = Path(output_dir) / 'logs' / 'batch_summary.json'

        import json
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return processed_reports

    def save_token_registry(self, path: str) -> None:
        """Save token registry to file."""
        self._anon_engine.save_registry(path)

    def load_token_registry(self, path: str) -> None:
        """Load token registry from file."""
        self._anon_engine._load_registry(path)

    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            'anonymization_engine': self._anon_engine.get_statistics(),
        }

    async def close(self) -> None:
        """Close resources."""
        await self.field_detector.close()
        self.ocr_processor.shutdown()
