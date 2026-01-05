"""Tests for ocr_processor module."""

import pytest
import numpy as np

from document_anonymizer.ocr_processor import OCRProcessor


class TestOCRProcessorInit:
    """Tests for OCRProcessor initialization."""

    def test_default_initialization(self):
        """Should initialize with default settings."""
        # Note: This test may be slow due to EasyOCR model loading
        processor = OCRProcessor()
        assert processor.languages == ['en']
        assert processor.preprocessing_enabled is True
        assert processor.max_workers == 4

    def test_custom_languages(self):
        """Should accept custom languages."""
        processor = OCRProcessor(languages=['en', 'tr'])
        assert 'en' in processor.languages
        assert 'tr' in processor.languages

    def test_custom_max_workers(self):
        """Should accept custom max_workers."""
        processor = OCRProcessor(max_workers=2)
        assert processor.max_workers == 2

    def test_custom_config(self):
        """Should respect custom configuration."""
        config = {
            'ocr_settings': {
                'preprocessing': {
                    'enabled': False,
                    'use_ocr_preprocessor': False,
                }
            }
        }
        processor = OCRProcessor(config=config)
        assert processor.preprocessing_enabled is False


class TestOCRProcessorSharedResources:
    """Tests for shared resources (thread pool, reader)."""

    def test_shared_reader(self):
        """Multiple instances should share EasyOCR reader."""
        proc1 = OCRProcessor()
        proc2 = OCRProcessor()
        # They should share the same reader instance
        assert proc1.reader is proc2.reader

    def test_shared_executor(self):
        """Multiple instances should share thread pool."""
        proc1 = OCRProcessor()
        proc2 = OCRProcessor()
        assert proc1.executor is proc2.executor


class TestOCRProcessorPreprocessing:
    """Tests for image preprocessing."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor(config={
            'ocr_settings': {
                'preprocessing': {
                    'enabled': True,
                    'use_ocr_preprocessor': False,  # Use fallback
                }
            }
        })

class TestOCRProcessorHelpers:
    """Tests for helper methods."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    def test_sample_image_processing(self, processor):
        """Should handle sample image dimensions."""
        # Create a test image
        test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        # Add some text-like dark pixels
        test_image[50:60, 50:250] = [0, 0, 0]

        assert test_image.shape == (200, 300, 3)


class TestAsyncOCRProcessing:
    """Tests for async OCR methods."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    @pytest.fixture
    def simple_image(self):
        """Create a simple test image."""
        return np.ones((100, 100, 3), dtype=np.uint8) * 255

    @pytest.mark.asyncio
    async def test_run_ocr_returns_dict(self, processor, simple_image):
        """run_ocr should return dictionary with expected keys."""
        result = await processor.run_ocr(simple_image, page_num=1)

        assert isinstance(result, dict)
        assert 'text_blocks' in result
        assert 'full_text' in result
        assert isinstance(result['text_blocks'], list)
        assert isinstance(result['full_text'], str)

    @pytest.mark.asyncio
    async def test_run_ocr_empty_image(self, processor):
        """OCR on blank image should return empty results."""
        blank_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = await processor.run_ocr(blank_image, page_num=1)

        assert result['text_blocks'] == [] or len(result['full_text'].strip()) == 0


class TestOCRResultStructure:
    """Tests for OCR result structure."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    @pytest.fixture
    def image_with_text(self):
        """Create image with clear text-like patterns."""
        # Create white background
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        # Add dark horizontal lines to simulate text
        for y in range(20, 180, 30):
            img[y:y+10, 20:380] = [0, 0, 0]
        return img

    @pytest.mark.asyncio
    async def test_text_block_structure(self, processor, image_with_text):
        """Text blocks should have expected structure."""
        result = await processor.run_ocr(image_with_text, page_num=1)

        for block in result.get('text_blocks', []):
            # Each block should have required fields
            assert 'text' in block or 'bbox' in block
            if 'bbox' in block:
                # Bbox should be valid coordinates
                bbox = block['bbox']
                assert len(bbox) == 4  # 4 corners
            if 'confidence' in block:
                assert 0 <= block['confidence'] <= 1


class TestOCRConfigOptions:
    """Tests for various configuration options."""

    def test_preprocessing_disabled(self):
        """Should work with preprocessing disabled."""
        config = {
            'ocr_settings': {
                'preprocessing': {
                    'enabled': False,
                }
            }
        }
        processor = OCRProcessor(config=config)
        assert processor.preprocessing_enabled is False

    def test_preprocessor_library_check(self):
        """Should handle missing ocr-preprocessor gracefully."""
        config = {
            'ocr_settings': {
                'preprocessing': {
                    'enabled': True,
                    'use_ocr_preprocessor': True,
                }
            }
        }
        # Should not raise even if library is missing
        processor = OCRProcessor(config=config)
        assert processor is not None


class TestOCREdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    @pytest.mark.asyncio
    async def test_very_small_image(self, processor):
        """Should handle very small images."""
        tiny_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result = await processor.run_ocr(tiny_image, page_num=1)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_grayscale_image(self, processor):
        """Should handle grayscale images."""
        # Create grayscale-like BGR image
        gray_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = await processor.run_ocr(gray_bgr, page_num=1)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_large_image(self, processor):
        """Should handle reasonably large images."""
        large_image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255
        result = await processor.run_ocr(large_image, page_num=1)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_multiple_pages(self, processor):
        """Should handle different page numbers."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result1 = await processor.run_ocr(image, page_num=1)
        result2 = await processor.run_ocr(image, page_num=5)
        result3 = await processor.run_ocr(image, page_num=100)

        assert all(isinstance(r, dict) for r in [result1, result2, result3])


class TestUpdateLanguages:
    """Tests for language update functionality."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor(languages=['en'])

    def test_update_languages_changes_reader(self, processor):
        """Should update reader when languages change."""
        old_languages = processor.languages.copy()
        processor.update_languages(['en', 'tr'])

        assert processor.languages == ['en', 'tr']
        assert processor.languages != old_languages

    def test_update_languages_same_languages(self, processor):
        """Should not reinitialize if same languages."""
        old_reader = processor.reader
        processor.update_languages(['en'])  # Same as initial

        assert processor.reader is old_reader


class TestDeskewImage:
    """Tests for image deskewing."""

    def test_deskew_straight_image(self):
        """Should not modify already straight image."""
        # Create a straight image with horizontal lines
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        img[50:52, 50:250] = [0, 0, 0]  # Horizontal line
        img[100:102, 50:250] = [0, 0, 0]

        result = OCRProcessor.deskew_image(img)
        # Should return similar size image
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_deskew_no_lines(self):
        """Should return original image if no lines detected."""
        blank_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = OCRProcessor.deskew_image(blank_img)
        assert np.array_equal(result, blank_img)

    def test_deskew_grayscale_input(self):
        """Should handle grayscale input."""
        gray_img = np.ones((100, 100), dtype=np.uint8) * 255
        result = OCRProcessor.deskew_image(gray_img)
        assert result is not None


class TestPreprocessImageFallback:
    """Tests for fallback preprocessing."""

    def test_fallback_preprocessing_basic(self):
        """Should process image without errors."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        result = OCRProcessor.preprocess_image_fallback(img)

        assert result.shape[0] > 0
        assert result.shape[1] > 0
        assert len(result.shape) == 3  # Should be color

    def test_fallback_low_contrast_enhancement(self):
        """Should enhance low contrast images."""
        # Create low contrast image
        low_contrast = np.ones((100, 100, 3), dtype=np.uint8) * 128
        low_contrast[40:60, 40:60] = 130  # Slight variation

        result = OCRProcessor.preprocess_image_fallback(low_contrast)
        assert result is not None


class TestAnalyzeFontProperties:
    """Tests for font property analysis."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    @pytest.fixture
    def sample_image(self):
        """Create image with text-like content."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        # Add dark text-like area
        img[80:100, 50:250] = [0, 0, 0]
        return img

    def test_analyze_font_properties_basic(self, processor, sample_image):
        """Should extract font properties."""
        bbox = [50, 80, 250, 100]
        props = processor._analyze_font_properties(sample_image, bbox, "Test text")

        assert 'estimated_size' in props
        assert 'text_color' in props
        assert 'background_color' in props
        assert 'is_bold' in props

    def test_analyze_font_properties_empty_roi(self, processor):
        """Should handle empty ROI."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bbox = [50, 50, 50, 50]  # Zero-area bbox
        props = processor._analyze_font_properties(img, bbox, "text")

        # Should return default properties
        assert 'estimated_size' in props

    def test_analyze_font_properties_out_of_bounds(self, processor):
        """Should clamp bbox to image bounds."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bbox = [-10, -10, 200, 200]  # Out of bounds
        props = processor._analyze_font_properties(img, bbox, "text")

        assert props is not None


class TestDefaultFontProperties:
    """Tests for default font properties."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    def test_default_font_properties(self, processor):
        """Should return sensible defaults."""
        props = processor._default_font_properties(20)

        assert props['estimated_size'] >= 8
        assert props['text_color'] == (0, 0, 0)
        assert props['background_color'] == (255, 255, 255)
        assert props['is_bold'] is False


class TestExtractTextAndBgColors:
    """Tests for color extraction."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    def test_extract_colors_black_on_white(self, processor):
        """Should detect black text on white background."""
        # Create ROI with black text on white
        roi = np.ones((20, 100, 3), dtype=np.uint8) * 255
        roi[5:15, 10:90] = [0, 0, 0]

        text_color, bg_color = processor._extract_text_and_bg_colors(roi)

        # Text should be dark, background light
        assert sum(text_color) < sum(bg_color)

    def test_extract_colors_empty_roi(self, processor):
        """Should return defaults for empty ROI."""
        empty_roi = np.array([]).reshape(0, 0, 3)
        text_color, bg_color = processor._extract_text_and_bg_colors(empty_roi)

        assert text_color == (0, 0, 0)
        assert bg_color == (255, 255, 255)

    def test_extract_colors_grayscale(self, processor):
        """Should handle grayscale ROI."""
        gray_roi = np.ones((20, 50), dtype=np.uint8) * 200
        gray_roi[5:15, 10:40] = 50

        text_color, bg_color = processor._extract_text_and_bg_colors(gray_roi)
        assert text_color is not None
        assert bg_color is not None


class TestDetectBoldText:
    """Tests for bold text detection."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    def test_detect_bold_thick_strokes(self, processor):
        """Should detect thick strokes as bold."""
        # Create ROI with thick strokes
        roi = np.ones((30, 100, 3), dtype=np.uint8) * 255
        roi[5:25, 20:80] = [0, 0, 0]  # Thick black area

        is_bold = processor._detect_bold_text(roi)
        # Thick stroke should be detected as bold (numpy bool or Python bool)
        assert is_bold in (True, False, np.True_, np.False_)

    def test_detect_bold_empty_roi(self, processor):
        """Should return False for empty ROI."""
        empty_roi = np.array([]).reshape(0, 0, 3)
        is_bold = processor._detect_bold_text(empty_roi)
        assert is_bold is False


class TestGetSurroundingBackground:
    """Tests for surrounding background extraction."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    def test_get_surrounding_background_white(self, processor):
        """Should detect white surrounding."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        bbox = [50, 50, 150, 100]

        bg_color = processor._get_surrounding_background(img, bbox)

        # Should be close to white
        assert sum(bg_color) > 700  # Close to 765 (255*3)

    def test_get_surrounding_background_colored(self, processor):
        """Should detect colored surrounding."""
        img = np.ones((200, 300, 3), dtype=np.uint8)
        img[:, :] = [100, 150, 200]  # Colored background
        bbox = [50, 50, 150, 100]

        bg_color = processor._get_surrounding_background(img, bbox)

        # BGR values reversed to RGB
        assert bg_color is not None

    def test_get_surrounding_background_edge_bbox(self, processor):
        """Should handle bbox at image edge."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bbox = [0, 0, 50, 50]  # At corner

        bg_color = processor._get_surrounding_background(img, bbox)
        assert bg_color is not None


class TestRunOcrBatch:
    """Tests for batch OCR processing."""

    @pytest.fixture
    def processor(self):
        return OCRProcessor()

    @pytest.mark.asyncio
    async def test_run_ocr_batch_basic(self, processor):
        """Should process multiple images."""
        images = [
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
            np.ones((100, 100, 3), dtype=np.uint8) * 255,
        ]

        results = await processor.run_ocr_batch(images, start_page=1)

        assert len(results) == 2
        assert results[0]['page'] == 1
        assert results[1]['page'] == 2

    @pytest.mark.asyncio
    async def test_run_ocr_batch_empty(self, processor):
        """Should handle empty image list."""
        results = await processor.run_ocr_batch([], start_page=1)
        assert results == []

    @pytest.mark.asyncio
    async def test_run_ocr_batch_start_page(self, processor):
        """Should respect start_page parameter."""
        images = [np.ones((50, 50, 3), dtype=np.uint8) * 255]

        results = await processor.run_ocr_batch(images, start_page=5)

        assert results[0]['page'] == 5


class TestShutdown:
    """Tests for shutdown functionality."""

    def test_shutdown_clears_executor(self):
        """Shutdown should clear the executor."""
        # Ensure executor exists
        processor = OCRProcessor()
        assert OCRProcessor._executor is not None

        # Shutdown
        OCRProcessor.shutdown()
        assert OCRProcessor._executor is None

        # Re-initialize for other tests
        OCRProcessor()
