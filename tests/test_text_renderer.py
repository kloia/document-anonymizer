"""Tests for text_renderer module."""

import numpy as np
import pytest

from document_anonymizer.text_renderer import TextRenderer
from document_anonymizer.anonymization_engine import AnonymizationEngine


class TestTextRendererInit:
    """Tests for TextRenderer initialization."""

    def test_default_initialization(self):
        """Should initialize with defaults."""
        renderer = TextRenderer()
        assert renderer.config == {}
        assert renderer.auto_scale is True
        assert renderer.padding == 5

    def test_custom_config(self):
        """Should use custom config values."""
        config = {
            "masking_strategy": {
                "font": {"auto_scale": False},
                "text_fields": {"padding": 10}
            }
        }
        renderer = TextRenderer(config=config)
        assert renderer.auto_scale is False
        assert renderer.padding == 10

    def test_injected_anonymization_engine(self):
        """Should accept injected anonymization engine."""
        engine = AnonymizationEngine(secret_key="test", persist_registry=False)
        renderer = TextRenderer(anonymization_engine=engine)
        assert renderer._anon_engine is engine


class TestDeterministicDummyGeneration:
    """Tests for dummy data generation."""

    @pytest.fixture
    def renderer(self):
        engine = AnonymizationEngine(secret_key="test", persist_registry=False)
        return TextRenderer(anonymization_engine=engine)

    def test_generate_deterministic_dummy_consistency(self, renderer):
        """Same input should produce same output."""
        result1 = renderer.generate_deterministic_dummy("John Smith", "person_name")
        result2 = renderer.generate_deterministic_dummy("John Smith", "person_name")
        assert result1 == result2

    def test_generate_deterministic_dummy_different_inputs(self, renderer):
        """Different inputs should produce different outputs."""
        result1 = renderer.generate_deterministic_dummy("John Smith", "person_name")
        result2 = renderer.generate_deterministic_dummy("Jane Doe", "person_name")
        assert result1 != result2

    def test_generate_deterministic_dummy_empty_input(self, renderer):
        """Empty input should return unchanged."""
        result = renderer.generate_deterministic_dummy("", "person_name")
        assert result == ""

    def test_generate_deterministic_dummy_whitespace(self, renderer):
        """Whitespace-only input should return unchanged."""
        result = renderer.generate_deterministic_dummy("   ", "person_name")
        assert result == "   "


class TestFontSizeEstimation:
    """Tests for font size estimation."""

    @pytest.fixture
    def renderer(self):
        return TextRenderer()

    def test_estimate_font_size_basic(self, renderer):
        """Should estimate font size from bbox."""
        bbox = (0, 0, 100, 30)  # 30 pixels high
        size = renderer.estimate_font_size(bbox)
        # Should be roughly 75% of height
        assert 8 <= size <= 30

    def test_estimate_font_size_with_text(self, renderer):
        """Should consider text length in estimation."""
        bbox = (0, 0, 200, 30)
        size_short = renderer.estimate_font_size(bbox, "Hi")
        size_long = renderer.estimate_font_size(bbox, "This is a much longer text")
        # Longer text should result in smaller font
        assert size_long <= size_short

    def test_estimate_font_size_min_max(self, renderer):
        """Should respect min/max bounds."""
        # Very small bbox
        small_bbox = (0, 0, 10, 5)
        size = renderer.estimate_font_size(small_bbox)
        assert size >= 8

        # Very large bbox
        large_bbox = (0, 0, 1000, 200)
        size = renderer.estimate_font_size(large_bbox)
        assert size <= 60


class TestMaskAndRender:
    """Tests for mask and render functionality."""

    @pytest.fixture
    def renderer(self):
        engine = AnonymizationEngine(secret_key="test", persist_registry=False)
        return TextRenderer(anonymization_engine=engine)

    @pytest.fixture
    def sample_image(self):
        """Create a white 200x300 BGR image."""
        return np.ones((200, 300, 3), dtype=np.uint8) * 255

    def test_mask_and_render_no_bbox(self, renderer, sample_image):
        """Should return original image if no bbox."""
        field = {"text": "test", "field_type": "person_name"}
        result, dummy = renderer.mask_and_render(sample_image, field)
        assert np.array_equal(result, sample_image)
        assert dummy == ""

    def test_mask_and_render_basic(self, renderer, sample_image):
        """Should mask and render text."""
        field = {
            "text": "John Smith",
            "original_text": "John Smith",
            "field_type": "person_name",
            "bbox": (50, 50, 150, 80),
            "page": 1,
            "detection_method": "llm"
        }
        result, dummy = renderer.mask_and_render(sample_image.copy(), field)

        # Result should be modified
        assert not np.array_equal(result, sample_image)
        # Dummy should be generated
        assert len(dummy) > 0

    def test_mask_and_render_with_label(self, renderer, sample_image):
        """Should preserve label in output."""
        field = {
            "text": "John Smith",
            "original_text": "John Smith",
            "label": "Name: ",
            "field_type": "person_name",
            "bbox": (50, 50, 200, 80),
            "page": 1,
            "detection_method": "llm"
        }
        result, dummy = renderer.mask_and_render(sample_image.copy(), field)

        # Dummy should be generated for the value, not the label
        assert len(dummy) > 0

    def test_mask_and_render_signature(self, renderer, sample_image):
        """Should handle signature fields differently."""
        field = {
            "text": "[SIGNATURE]",
            "field_type": "signature",
            "bbox": (50, 50, 150, 120),
            "detection_method": "visual_detection"
        }
        result, dummy = renderer.mask_and_render(sample_image.copy(), field)

        # Signature should use contour masking, dummy should be empty
        assert dummy == ""


class TestFontLoading:
    """Tests for font loading functionality."""

    @pytest.fixture
    def renderer(self):
        return TextRenderer()

    def test_load_styled_font_normal(self, renderer):
        """Should load normal font."""
        font = renderer._load_styled_font(12, is_bold=False)
        assert font is not None

    def test_load_styled_font_bold(self, renderer):
        """Should load bold font."""
        font = renderer._load_styled_font(12, is_bold=True)
        assert font is not None

    def test_load_styled_font_caching(self, renderer):
        """Should cache loaded fonts."""
        font1 = renderer._load_styled_font(12, is_bold=False)
        font2 = renderer._load_styled_font(12, is_bold=False)
        assert font1 is font2  # Same cached instance

    def test_load_styled_font_different_sizes(self, renderer):
        """Different sizes should produce different fonts."""
        font12 = renderer._load_styled_font(12, is_bold=False)
        font20 = renderer._load_styled_font(20, is_bold=False)
        # Both should be loaded (different cache keys)
        assert ('normal', 12) in renderer._font_cache
        assert ('normal', 20) in renderer._font_cache


class TestMaskAndRenderEdgeCases:
    """Tests for mask_and_render edge cases."""

    @pytest.fixture
    def renderer(self):
        engine = AnonymizationEngine(secret_key="test", persist_registry=False)
        return TextRenderer(anonymization_engine=engine)

    @pytest.fixture
    def sample_image(self):
        return np.ones((200, 300, 3), dtype=np.uint8) * 255

    def test_mask_stamp_field(self, renderer, sample_image):
        """Should handle stamp fields with contour masking."""
        field = {
            "text": "[STAMP]",
            "field_type": "stamp",
            "bbox": (50, 50, 150, 120),
            "detection_method": "visual_detection"
        }
        result, dummy = renderer.mask_and_render(sample_image.copy(), field)
        assert dummy == ""

    def test_mask_visual_detection(self, renderer, sample_image):
        """Should use contour masking for visual detection."""
        field = {
            "text": "Visual element",
            "field_type": "unknown",
            "bbox": (50, 50, 150, 100),
            "detection_method": "visual_detection"
        }
        result, dummy = renderer.mask_and_render(sample_image.copy(), field)
        assert dummy == ""

    def test_mask_with_font_properties(self, renderer, sample_image):
        """Should use font properties from field."""
        field = {
            "text": "John Smith",
            "original_text": "John Smith",
            "field_type": "person_name",
            "bbox": (50, 50, 200, 80),
            "page": 1,
            "detection_method": "llm",
            "font_properties": {
                "estimated_size": 14,
                "is_bold": True,
                "background_color": (255, 255, 255)
            }
        }
        result, dummy = renderer.mask_and_render(sample_image.copy(), field)
        assert len(dummy) > 0

    def test_mask_list_bbox(self, renderer, sample_image):
        """Should handle list bbox format."""
        field = {
            "text": "Test",
            "original_text": "Test",
            "field_type": "person_name",
            "bbox": [50, 50, 150, 80],
            "page": 1,
            "detection_method": "llm"
        }
        result, dummy = renderer.mask_and_render(sample_image.copy(), field)
        assert result is not None


class TestFontSizeConstraints:
    """Tests for font size estimation constraints."""

    @pytest.fixture
    def renderer(self):
        return TextRenderer()

    def test_very_small_bbox(self, renderer):
        """Should enforce minimum font size."""
        bbox = (0, 0, 5, 5)
        size = renderer.estimate_font_size(bbox)
        assert size >= 8

    def test_very_large_bbox(self, renderer):
        """Should enforce maximum font size."""
        bbox = (0, 0, 1000, 500)
        size = renderer.estimate_font_size(bbox)
        assert size <= 60
