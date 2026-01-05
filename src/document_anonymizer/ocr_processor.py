"""Async OCR processor using EasyOCR with optional preprocessing."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np

# Import ocr-preprocessor for enhanced preprocessing
try:
    from ocr_preprocessor import OCRPreprocessor, Pipeline

    HAS_OCR_PREPROCESSOR = True
except ImportError:
    HAS_OCR_PREPROCESSOR = False
    OCRPreprocessor = None
    Pipeline = None

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Async EasyOCR processor with preprocessing and font analysis."""

    _executor: Optional[ThreadPoolExecutor] = None
    _readers: Dict[str, easyocr.Reader] = {}

    def __init__(
        self,
        config: Optional[Dict] = None,
        max_workers: int = 4,
        languages: Optional[List[str]] = None,
    ):
        """Initialize OCR processor."""
        self.config = config or {}
        self.max_workers = max_workers

        # OCR settings
        ocr_settings = self.config.get("ocr_settings", {})
        preprocessing = ocr_settings.get("preprocessing", {})
        self.preprocessing_enabled = preprocessing.get("enabled", True)
        self.use_ocr_preprocessor = preprocessing.get("use_ocr_preprocessor", True)
        self.preprocessing_pipeline = preprocessing.get("pipeline", "full")

        # Language settings - from config, parameter, or auto-detect later
        config_languages = ocr_settings.get("languages")
        self.languages = config_languages or languages or ["en"]

        # Initialize ocr-preprocessor if available
        self._preprocessor: Optional[OCRPreprocessor] = None
        if HAS_OCR_PREPROCESSOR and self.use_ocr_preprocessor:
            self._preprocessor = OCRPreprocessor(
                min_width=preprocessing.get("min_width", 1000),
                max_width=preprocessing.get("max_width", 3000),
            )
            logger.debug("OCR Preprocessor initialized")
        elif self.use_ocr_preprocessor:
            logger.warning("ocr-preprocessor not installed, using fallback preprocessing")

        # Initialize thread pool if not exists
        if OCRProcessor._executor is None:
            OCRProcessor._executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="ocr_worker"
            )

        self.executor = OCRProcessor._executor
        self.reader = self._get_reader(self.languages)

    def _get_reader(self, languages: List[str]) -> easyocr.Reader:
        """Get or create cached EasyOCR reader for given languages."""
        cache_key = ",".join(sorted(languages))

        if cache_key not in OCRProcessor._readers:
            logger.info(f"Initializing EasyOCR with languages: {languages}")
            OCRProcessor._readers[cache_key] = easyocr.Reader(languages, gpu=True, verbose=False)

        return OCRProcessor._readers[cache_key]

    def update_languages(self, languages: List[str]) -> None:
        """
        Update OCR languages and reinitialize reader.

        Args:
            languages: List of language codes (e.g., ['en', 'tr'])
        """
        if set(languages) != set(self.languages):
            logger.info(f"Updating OCR languages: {self.languages} → {languages}")
            self.languages = languages
            self.reader = self._get_reader(languages)

    def preprocess_with_ocr_preprocessor(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image using ocr-preprocessor library.

        Args:
            image: OpenCV format image (BGR)

        Returns:
            Preprocessed image
        """
        if self._preprocessor is None:
            return image

        try:
            # Convert OpenCV BGR to bytes
            _, img_bytes = cv2.imencode(".png", image)
            img_bytes = img_bytes.tobytes()

            # Get pipeline
            if self.preprocessing_pipeline == "minimal":
                pipeline = Pipeline.MINIMAL
            elif self.preprocessing_pipeline == "fast":
                pipeline = Pipeline.FAST
            else:
                pipeline = Pipeline.FULL

            # Process image
            processed_bytes = self._preprocessor.process_image(img_bytes, pipeline=pipeline)

            # Convert back to OpenCV format
            nparr = np.frombuffer(processed_bytes, np.uint8)
            processed = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            logger.debug(f"Image preprocessed with ocr-preprocessor ({pipeline.name})")
            return processed

        except Exception as e:
            logger.warning(f"OCR preprocessor failed, using original image: {e}")
            return image

    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """
        Correct image skew (rotation) for better OCR accuracy.

        Uses Hough Line Transform to detect dominant lines and correct rotation.

        Args:
            image: Image (grayscale or color)

        Returns:
            Deskewed image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return image

        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return image

        # Use median angle to avoid outliers
        median_angle = np.median(angles)

        # Only correct if skew is significant but not too large
        if abs(median_angle) < 0.5 or abs(median_angle) > 10:
            return image

        # Get image dimensions
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Calculate new image bounds
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255,
        )

        logger.debug(f"Image deskewed by {median_angle:.2f} degrees")
        return rotated

    @staticmethod
    def preprocess_image_fallback(image: np.ndarray) -> np.ndarray:
        """
        Fallback preprocessing when ocr-preprocessor is not available.

        Args:
            image: OpenCV format image (BGR)

        Returns:
            Preprocessed image
        """
        result = image.copy()

        # Deskew
        result = OCRProcessor.deskew_image(result)

        # Check contrast
        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result

        std_brightness = np.std(gray)

        # Apply CLAHE for low contrast images
        if std_brightness < 40:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return result

    def _analyze_font_properties(self, image: np.ndarray, bbox: List[int], text: str) -> Dict:
        """
        Analyze font properties from text region.

        Args:
            image: OpenCV format image (BGR)
            bbox: [x1, y1, x2, y2] coordinates
            text: Detected text content

        Returns:
            Dictionary with font properties
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Ensure valid bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # Extract region of interest
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return self._default_font_properties(y2 - y1)

        # Estimate font size from height
        text_height = y2 - y1
        estimated_font_size = int(text_height * 0.85)

        # Analyze text and background colors
        text_color, bg_color = self._extract_text_and_bg_colors(roi)

        # Detect bold text
        is_bold = self._detect_bold_text(roi)

        # Character spacing
        if len(text) > 1 and text_height > 0:
            char_width = (x2 - x1) / len(text)
            char_spacing = "normal" if char_width < text_height * 0.7 else "wide"
        else:
            char_spacing = "normal"

        # Get surrounding background
        surrounding_bg = self._get_surrounding_background(image, bbox)

        return {
            "estimated_size": max(8, min(estimated_font_size, 72)),
            "text_color": text_color,
            "background_color": bg_color,
            "surrounding_background": surrounding_bg,
            "is_bold": is_bold,
            "char_spacing": char_spacing,
            "text_height": text_height,
            "text_width": x2 - x1,
        }

    def _default_font_properties(self, height: int) -> Dict:
        """Return default font properties."""
        return {
            "estimated_size": max(8, int(height * 0.85)),
            "text_color": (0, 0, 0),
            "background_color": (255, 255, 255),
            "surrounding_background": (255, 255, 255),
            "is_bold": False,
            "char_spacing": "normal",
            "text_height": height,
            "text_width": 0,
        }

    def _extract_text_and_bg_colors(
        self, roi: np.ndarray
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Extract text and background colors from ROI."""
        if roi.size == 0:
            return (0, 0, 0), (255, 255, 255)

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)

        if white_pixels > black_pixels:
            text_mask = binary == 0
            bg_mask = binary == 255
        else:
            text_mask = binary == 255
            bg_mask = binary == 0

        if len(roi.shape) == 3:
            text_pixels = roi[text_mask]
            if len(text_pixels) > 0:
                text_color = np.mean(text_pixels, axis=0).astype(int)
                text_color = tuple(text_color[::-1].tolist())
            else:
                text_color = (0, 0, 0)

            bg_pixels = roi[bg_mask]
            if len(bg_pixels) > 0:
                bg_color = np.mean(bg_pixels, axis=0).astype(int)
                bg_color = tuple(bg_color[::-1].tolist())
            else:
                bg_color = (255, 255, 255)
        else:
            text_val = int(np.mean(gray[text_mask])) if np.any(text_mask) else 0
            bg_val = int(np.mean(gray[bg_mask])) if np.any(bg_mask) else 255
            text_color = (text_val, text_val, text_val)
            bg_color = (bg_val, bg_val, bg_val)

        return text_color, bg_color

    def _detect_bold_text(self, roi: np.ndarray) -> bool:
        """Detect if text is bold by analyzing stroke thickness."""
        if roi.size == 0:
            return False

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        h, w = gray.shape

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        non_zero = dist_transform[dist_transform > 0]
        if len(non_zero) == 0:
            return False

        avg_stroke = np.mean(non_zero)
        relative_stroke = avg_stroke / h if h > 0 else 0

        return avg_stroke > 2.5 or relative_stroke > 0.08

    def _get_surrounding_background(
        self, image: np.ndarray, bbox: List[int], border_width: int = 15
    ) -> Tuple[int, int, int]:
        """Get background color from area surrounding the bbox."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        border_pixels = []

        # Sample from borders
        regions = [
            (max(0, y1 - border_width), max(0, y1), max(0, x1), min(w, x2)),  # Top
            (min(h, y2), min(h, y2 + border_width), max(0, x1), min(w, x2)),  # Bottom
            (max(0, y1), min(h, y2), max(0, x1 - border_width), max(0, x1)),  # Left
            (max(0, y1), min(h, y2), min(w, x2), min(w, x2 + border_width)),  # Right
        ]

        for y1r, y2r, x1r, x2r in regions:
            if y2r > y1r and x2r > x1r:
                region = image[y1r:y2r, x1r:x2r]
                if region.size > 0:
                    border_pixels.extend(region.reshape(-1, 3).tolist())

        if border_pixels:
            avg_color = np.mean(border_pixels, axis=0).astype(int)
            return tuple(avg_color[::-1].tolist())
        else:
            return (255, 255, 255)

    def _run_ocr_sync(self, image: np.ndarray, page_num: int) -> Dict:
        """
        Synchronous OCR operation (runs in thread pool).

        Args:
            image: OpenCV format image
            page_num: Page number

        Returns:
            OCR results dictionary
        """
        logger.debug(f"Running OCR for page {page_num}...")

        # Store original dimensions for coordinate scaling
        orig_h, orig_w = image.shape[:2]

        # Preprocessing
        if self.preprocessing_enabled:
            if self._preprocessor is not None:
                processed = self.preprocess_with_ocr_preprocessor(image)
            else:
                processed = self.preprocess_image_fallback(image)
        else:
            processed = image

        proc_h, proc_w = processed.shape[:2]

        # Calculate scale factors to convert coordinates back to original
        scale_x = orig_w / proc_w if proc_w > 0 else 1.0
        scale_y = orig_h / proc_h if proc_h > 0 else 1.0

        if scale_x != 1.0 or scale_y != 1.0:
            logger.debug(f"Page {page_num}: Scale factors x={scale_x:.3f}, y={scale_y:.3f}")

        try:
            # Run EasyOCR
            results = self.reader.readtext(processed)

            text_blocks = []
            full_text_parts = []
            block_counter = 0

            for detection in results:
                bbox_points, text, confidence = detection

                # Convert polygon to rectangle (in processed coordinates)
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]

                proc_x1 = int(min(x_coords))
                proc_y1 = int(min(y_coords))
                proc_x2 = int(max(x_coords))
                proc_y2 = int(max(y_coords))

                if not text or len(text.strip()) < 2:
                    continue

                # Analyze font properties on processed image (for consistency)
                font_props = self._analyze_font_properties(
                    processed, [proc_x1, proc_y1, proc_x2, proc_y2], text.strip()
                )

                # Scale font size estimate to original dimensions
                if "estimated_size" in font_props:
                    font_props["estimated_size"] = int(font_props["estimated_size"] * scale_y)

                # Scale bbox to original image coordinates
                orig_x1 = int(proc_x1 * scale_x)
                orig_y1 = int(proc_y1 * scale_y)
                orig_x2 = int(proc_x2 * scale_x)
                orig_y2 = int(proc_y2 * scale_y)

                # Scale polygon to original coordinates
                orig_polygon = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in bbox_points]

                block_counter += 1
                text_blocks.append(
                    {
                        "block_id": f"block_{page_num}_{block_counter}",
                        "text": text.strip(),
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "confidence": round(confidence, 3),
                        "polygon": orig_polygon,
                        "font_properties": font_props,
                        "page": page_num,
                    }
                )

                full_text_parts.append(text.strip())

            # Sort by position
            text_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

            result = {
                "page": page_num,
                "text_blocks": text_blocks,
                "full_text": " ".join(full_text_parts),
                "image_size": {"width": orig_w, "height": orig_h},
            }

            logger.debug(f"Page {page_num}: {len(text_blocks)} text blocks detected")
            return result

        except Exception as e:
            logger.error(f"OCR error on page {page_num}: {e}")
            return {"page": page_num, "text_blocks": [], "full_text": "", "error": str(e)}

    async def run_ocr(self, image: np.ndarray, page_num: int = 1) -> Dict:
        """
        Run OCR asynchronously using thread pool.

        Args:
            image: OpenCV format image (BGR)
            page_num: Page number

        Returns:
            OCR results with text_blocks list
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._run_ocr_sync, image, page_num)

    async def run_ocr_batch(self, images: List[np.ndarray], start_page: int = 1) -> List[Dict]:
        """
        Run OCR on multiple images in parallel.

        Args:
            images: List of images
            start_page: Starting page number

        Returns:
            List of OCR results
        """
        tasks = [self.run_ocr(img, start_page + i) for i, img in enumerate(images)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Page {start_page + i} OCR failed: {result}")
                processed_results.append(
                    {
                        "page": start_page + i,
                        "text_blocks": [],
                        "full_text": "",
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    @classmethod
    def shutdown(cls):
        """Shutdown the thread pool executor."""
        if cls._executor:
            cls._executor.shutdown(wait=True)
            cls._executor = None
            logger.info("OCR thread pool shut down")
