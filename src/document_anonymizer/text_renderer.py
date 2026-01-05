"""Dummy text generation and rendering module."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .anonymization_engine import AnonymizationEngine, create_anonymization_engine
from .image_masker import ImageMasker
from .utils import is_signature_or_stamp, validate_bbox

logger = logging.getLogger(__name__)


class TextRenderer:
    """Dummy text generator and renderer with font matching support."""

    FONT_PATHS = {
        'normal': [
            'config/fonts/Arial.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/Arial.ttf',
            '/Library/Fonts/Arial.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            'C:/Windows/Fonts/arial.ttf',
        ],
        'bold': [
            'config/fonts/Arial-Bold.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            '/Library/Fonts/Arial Bold.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            'C:/Windows/Fonts/arialbd.ttf',
        ]
    }

    def __init__(
        self,
        config: Optional[Dict] = None,
        anonymization_engine: Optional[AnonymizationEngine] = None,
        image_masker: Optional["ImageMasker"] = None
    ):
        """Initialize text renderer."""
        self.config = config or {}

        font_config = self.config.get('masking_strategy', {}).get('font', {})
        self.default_font_path = font_config.get('default_path', 'config/fonts/Arial.ttf')
        self.default_font_size = font_config.get('default_size', 20)
        self.auto_scale = font_config.get('auto_scale', True)

        text_config = self.config.get('masking_strategy', {}).get('text_fields', {})
        self.default_text_color = tuple(text_config.get('text_color', [0, 0, 0]))
        self.default_bg_color = tuple(text_config.get('background_color', [255, 255, 255]))
        self.padding = text_config.get('padding', 5)

        self._font_cache = {}

        # Dependency injection for anonymization engine
        if anonymization_engine:
            self._anon_engine = anonymization_engine
        else:
            registry_path = self.config.get('anonymization', {}).get('registry_path')
            secret_key = self.config.get('anonymization', {}).get('secret_key')
            self._anon_engine = create_anonymization_engine(
                secret_key=secret_key,
                registry_path=registry_path
            )

        # Dependency injection for image masker
        self._image_masker = image_masker or ImageMasker(self.config)

        logger.debug("TextRenderer initialized")

    def generate_deterministic_dummy(
        self,
        original_text: str,
        field_type: str,
        document_id: Optional[str] = None,
        field_context: Optional[str] = None
    ) -> str:
        """
        Generate deterministic dummy data for original text.

        Args:
            original_text: Original sensitive data
            field_type: Field type
            document_id: Optional document identifier
            field_context: Optional context info (page, detection method)

        Returns:
            Dummy text (anonymized token)
        """
        if not original_text or not original_text.strip():
            return original_text

        return self._anon_engine.anonymize(
            original_text=original_text,
            field_type=field_type,
            document_id=document_id,
            context=field_context
        )

    def mask_and_render(
        self,
        image: np.ndarray,
        field: Dict,
        font_properties: Optional[Dict] = None,
        hash_mapping: Optional[Dict] = None,
        document_id: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Mask field area and render dummy text.

        Supports label preservation: if field has 'label', only the
        sensitive_value is anonymized and label is preserved.

        Args:
            image: OpenCV image (BGR)
            field: Field information dict (may include 'label' for preservation)
            font_properties: Font properties (optional)
            hash_mapping: Mapping cache
            document_id: Document identifier for audit trail

        Returns:
            (Processed image, dummy text) tuple
        """
        bbox = field.get('bbox')
        field_type = field.get('field_type', '')

        if not bbox:
            return image, ''

        if is_signature_or_stamp(field):
            masked = self._image_masker.mask_signature_stamp_contour(image.copy(), bbox)
            return masked, ''

        # Handle label preservation
        label = field.get('label')  # e.g., "Phone: "
        if label:
            # Anonymize only the sensitive value, preserve label
            sensitive_value = field.get('original_text', '') or field.get('text', '')
        else:
            # No label - anonymize entire text
            sensitive_value = field.get('original_text', '') or field.get('text', '')

        if font_properties is None:
            font_properties = field.get('font_properties', {})

        bg_color = font_properties.get('surrounding_background', self.default_bg_color)
        text_color = font_properties.get('text_color', self.default_text_color)
        is_bold = font_properties.get('is_bold', False)
        estimated_size = font_properties.get('estimated_size', None)

        # Build field context for audit trail
        page_num = field.get('page', 0)
        detection_method = field.get('detection_method', 'unknown')
        field_context = f"page:{page_num}|type:{field_type}|method:{detection_method}"

        # Generate dummy for the sensitive value only
        dummy_value = self.generate_deterministic_dummy(
            sensitive_value, field_type,
            document_id=document_id,
            field_context=field_context
        )

        # Combine label + dummy_value for final text
        if label:
            render_text = f"{label}{dummy_value}"
            logger.debug(f"Label preserved: '{label}' + '{dummy_value}'")
        else:
            render_text = dummy_value

        if hash_mapping is not None and sensitive_value:
            hash_mapping[sensitive_value] = dummy_value

        masked = self._mask_with_background(image.copy(), bbox, bg_color)

        if estimated_size and self.auto_scale:
            font_size = self._fit_text_to_bbox(render_text, bbox, estimated_size, is_bold)
        else:
            font_size = self.estimate_font_size(bbox, sensitive_value)
            font_size = self._fit_text_to_bbox(render_text, bbox, font_size, is_bold)

        result = self._render_text_with_style(
            masked, render_text, bbox, font_size, text_color, is_bold
        )

        return result, dummy_value

    def _mask_with_background(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        bg_color: Tuple[int, int, int],
        safe_padding: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """Mask area with detected background color."""
        x1, y1, x2, y2 = validate_bbox(bbox, image.shape)

        if safe_padding is not None:
            left_pad, top_pad, right_pad, bottom_pad = safe_padding
        else:
            left_pad = top_pad = right_pad = bottom_pad = 0

        x1 = max(0, x1 - left_pad)
        y1 = max(0, y1 - top_pad)
        x2 = min(image.shape[1], x2 + right_pad)
        y2 = min(image.shape[0], y2 + bottom_pad)

        bgr_color = bg_color[::-1] if len(bg_color) == 3 else bg_color
        cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, -1)

        return image

    def _render_text_with_style(
        self,
        image: np.ndarray,
        text: str,
        bbox: Tuple[int, int, int, int],
        font_size: int,
        text_color: Tuple[int, int, int],
        is_bold: bool = False
    ) -> np.ndarray:
        """Render text with specific styling."""
        if not text or not text.strip():
            return image

        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            return image

        try:
            font = self._load_styled_font(max(6, font_size), is_bold)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_height = text_bbox[3] - text_bbox[1]
            except Exception:
                text_height = font_size

            box_height = y2 - y1
            text_x = x1 + 3

            if box_height > text_height:
                text_y = y1 + (box_height - text_height) // 2
            else:
                text_y = y1 + 2

            if '\n' in text:
                lines = text.split('\n')
                line_height = font_size + 4
                current_y = y1 + 3

                for line in lines:
                    if current_y + line_height > y2:
                        break
                    if line.strip():
                        draw.text((text_x, current_y), line, font=font, fill=text_color)
                    current_y += line_height
            else:
                draw.text((text_x, text_y), text, font=font, fill=text_color)

            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"Text rendering error: {e}")
            return image

    def _load_styled_font(
        self,
        size: int,
        is_bold: bool = False
    ) -> ImageFont.FreeTypeFont:
        """Load font with appropriate style."""
        style = 'bold' if is_bold else 'normal'
        cache_key = (style, size)

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font_paths = self.FONT_PATHS.get(style, self.FONT_PATHS['normal'])

        font = None
        for font_path in font_paths:
            try:
                if Path(font_path).exists():
                    font = ImageFont.truetype(font_path, size)
                    break
            except Exception:
                continue

        if font is None:
            try:
                font = ImageFont.truetype(self.default_font_path, size)
            except Exception:
                font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def _fit_text_to_bbox(
        self,
        text: str,
        bbox: Tuple[int, int, int, int],
        initial_size: int,
        is_bold: bool = False
    ) -> int:
        """Adjust font size to fit text within bbox."""
        if not text or not text.strip():
            return max(6, initial_size)

        x1, y1, x2, y2 = bbox
        box_width = max(1, x2 - x1 - 10)
        box_height = max(1, y2 - y1)

        if box_width <= 0 or box_height <= 0:
            return max(6, initial_size)

        temp_img = Image.new('RGB', (100, 100))
        draw = ImageDraw.Draw(temp_img)

        min_size = 6
        max_size = min(initial_size + 10, 72)
        best_size = max(min_size, min(initial_size, max_size))

        try:
            for size in range(max_size, min_size - 1, -1):
                try:
                    font = self._load_styled_font(size, is_bold)
                    if not text.strip():
                        break
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    if text_width <= box_width and text_height <= box_height:
                        best_size = size
                        break
                except (OSError, ZeroDivisionError):
                    continue
        except Exception:
            pass

        return max(min_size, best_size)

    def estimate_font_size(
        self,
        bbox: Tuple[int, int, int, int],
        original_text: Optional[str] = None
    ) -> int:
        """Estimate font size from bounding box."""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1

        height_based = int(height * 0.75)

        if original_text and len(original_text) > 0:
            char_count = len(original_text)
            width_based = int(width / (char_count * 0.55))
            estimated = int(height_based * 0.7 + width_based * 0.3)
        else:
            estimated = height_based

        return max(8, min(estimated, 60))

