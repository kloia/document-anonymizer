"""
LLM-based Sensitive Field Detection Module

Uses vision-language models for sensitive field detection.
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class LLMDetector:
    """
    LLM-based sensitive field detector.

    Uses vision-language models for:
    - Sensitive field detection
    - Visual element (signature/stamp) detection
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LLM detector.

        Args:
            config: Configuration dictionary containing LLM settings
        """
        self.config = config or {}
        llm_config = self.config.get("llm", {})

        # API settings: config > env var > default
        # This allows both programmatic configuration and environment variables
        self.api_url = llm_config.get("api_url") or os.getenv("LLM_API_URL", "")
        self.api_key = llm_config.get("api_key") or os.getenv("LLM_API_KEY", "")
        self.model = llm_config.get("model") or os.getenv("LLM_MODEL_VISION", "gpt-4o")

        # Retry settings
        self.max_retries = llm_config.get("max_retries", 3)
        self.timeout = llm_config.get("timeout", 120)
        self.retry_delay = llm_config.get("retry_delay", 2)

        # Load prompt
        self._prompt = self._load_prompt()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        logger.debug(f"LLMDetector initialized (model: {self.model})")

    def _load_prompt(self) -> str:
        """Load the sensitive field detector prompt."""
        prompt_path = Path(__file__).parent / "prompts" / "sensitive_field_detector.md"
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        logger.warning("Prompt file not found: sensitive_field_detector.md")
        return ""

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=10),
            )
        return self._client

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64."""
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")  # type: ignore[arg-type]

    async def detect_document_language(self, image: np.ndarray) -> Dict:
        """Detect document language. Returns {'languages': [...], 'locale': '...'}."""
        default_result = {"languages": ["en"], "locale": "en_US"}

        if not self.api_key:
            logger.warning("No LLM API key configured, using defaults")
            return default_result

        try:
            image_b64 = self._encode_image(image)
            prompt = """Analyze this document and identify its language(s).

Return a JSON object with:
1. "languages": Array of ISO 639-1 codes for OCR (e.g., ["en", "tr"])
2. "locale": Primary Faker locale for generating realistic dummy data (e.g., "tr_TR")

Common language codes: en, tr, de, fr, es, it, pt, nl, ru, ar, zh, ja, ko
Common locales: en_US, tr_TR, de_DE, fr_FR, es_ES, it_IT, pt_PT, nl_NL, ru_RU, ar_SA, zh_CN, ja_JP, ko_KR

Example responses:
- English document: {"languages": ["en"], "locale": "en_US"}
- Turkish document: {"languages": ["en", "tr"], "locale": "tr_TR"}
- German document: {"languages": ["en", "de"], "locale": "de_DE"}

Return only the JSON object, nothing else."""

            response = await self._call_llm_with_image(image_b64, prompt)

            # Parse response
            import json
            import re

            # Extract JSON object
            match = re.search(r"\{[\s\S]*?\}", response)
            if match:
                result = json.loads(match.group())
                languages = result.get("languages", ["en"])
                locale = result.get("locale", "en_US")

                # Ensure 'en' is always included in languages
                if "en" not in languages:
                    languages.insert(0, "en")

                logger.info(f"LLM detected languages: {languages}, locale: {locale}")
                return {"languages": languages, "locale": locale}

            logger.warning(f"Could not parse language response: {response}")
            return default_result

        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return default_result

    async def detect_all(self, image: np.ndarray, ocr_results: Optional[List[Dict]] = None) -> Dict:
        """Unified detection of text and visual sensitive elements."""
        empty_result: Dict[str, List] = {"text_detections": [], "visual_detections": []}

        if not self.api_key:
            logger.warning("No LLM API key configured")
            return empty_result

        try:
            image_b64 = self._encode_image(image)
            if not self._prompt:
                logger.warning("Unified detector prompt not found")
                return empty_result

            prompt = self._prompt

            # Add OCR blocks
            if ocr_results:
                import json

                ocr_blocks = [
                    {
                        "block_id": block.get("block_id", f"block_{i}"),
                        "text": block.get("text", ""),
                        "bbox": {
                            "x1": block.get("bbox", [0, 0, 0, 0])[0],
                            "y1": block.get("bbox", [0, 0, 0, 0])[1],
                            "x2": block.get("bbox", [0, 0, 0, 0])[2],
                            "y2": block.get("bbox", [0, 0, 0, 0])[3],
                        },
                    }
                    for i, block in enumerate(ocr_results[:100])
                ]
                ocr_json = json.dumps({"ocr_blocks": ocr_blocks}, ensure_ascii=False, indent=2)
                prompt += f"\n\n### OCR EXTRACTED BLOCKS\n```json\n{ocr_json}\n```"

            response = await self._call_llm_with_image(image_b64, prompt)
            return self._parse_unified_response(response, ocr_results or [])

        except Exception as e:
            logger.error(f"Unified detection error: {e}")
            return empty_result

    def _parse_unified_response(self, response: str, ocr_results: List[Dict]) -> Dict:
        """Parse unified detector LLM response."""
        import json
        import re

        result: Dict[str, List] = {"text_detections": [], "visual_detections": []}

        try:
            # Extract JSON object
            json_match = re.search(r'\{[\s\S]*"text_detections"[\s\S]*\}', response)
            if not json_match:
                # Try alternative pattern
                json_match = re.search(r'\{[\s\S]*"visual_detections"[\s\S]*\}', response)

            if json_match:
                data = json.loads(json_match.group())

                # Create OCR lookup for bbox retrieval
                ocr_lookup = {b.get("block_id"): b for b in ocr_results}

                # Parse text detections
                for td in data.get("text_detections", []):
                    block_id = td.get("block_id", "")
                    ocr_block = ocr_lookup.get(block_id, {})

                    detection = {
                        "block_id": block_id,
                        "full_text": td.get("full_text", ""),
                        "label": td.get("label"),  # Can be None
                        "sensitive_value": td.get("sensitive_value", ""),
                        "field_type": td.get("category", "unknown"),
                        "confidence": td.get("confidence", 0.5),
                        "risk_level": td.get("risk_level", "MEDIUM"),
                        "reasoning": td.get("reasoning", ""),
                        "bbox": ocr_block.get("bbox"),
                        "font_properties": ocr_block.get("font_properties", {}),
                        "detection_method": "llm_unified",
                    }
                    result["text_detections"].append(detection)

                # Parse visual detections
                for vd in data.get("visual_detections", []):
                    bbox_data = vd.get("bbox", {})
                    if isinstance(bbox_data, dict):
                        bbox = [
                            bbox_data.get("x1", 0),
                            bbox_data.get("y1", 0),
                            bbox_data.get("x2", 0),
                            bbox_data.get("y2", 0),
                        ]
                    else:
                        bbox = bbox_data if bbox_data else [0, 0, 0, 0]

                    detection = {
                        "element_id": vd.get(
                            "element_id", f"visual_{len(result['visual_detections'])}"
                        ),
                        "field_type": vd.get("type", "signature").lower(),
                        "bbox": bbox,
                        "confidence": vd.get("confidence", 0.8),
                        "description": vd.get("description", ""),
                        "is_visual": True,
                        "detection_method": "llm_unified_visual",
                    }
                    result["visual_detections"].append(detection)

                logger.info(
                    f"Unified detection: {len(result['text_detections'])} text, "
                    f"{len(result['visual_detections'])} visual"
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse unified response: {e}")

        return result

    async def _call_llm_with_image(self, image_b64: str, prompt: str) -> str:
        """
        Call LLM API with image and prompt.

        Args:
            image_b64: Base64 encoded image
            prompt: Text prompt

        Returns:
            LLM response text
        """
        client = await self._get_client()

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.api_url}/v1/chat/completions", headers=headers, json=payload
                )
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                body = e.response.text[:500] if e.response else ""
                logger.warning(f"LLM API error (attempt {attempt + 1}): {e} | body: {body}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

        return ""

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
