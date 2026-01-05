"""Shared pytest fixtures and configuration."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image for testing."""
    # Create a simple 100x100 white image with some text-like content
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Add some dark pixels to simulate text
    image[20:30, 10:90] = [0, 0, 0]  # Horizontal line
    image[40:50, 10:90] = [0, 0, 0]  # Another line
    return image


@pytest.fixture
def sample_ocr_result() -> list[dict[str, Any]]:
    """Sample OCR result for testing."""
    return [
        {
            "text": "John Smith",
            "bbox": [[10, 20], [90, 20], [90, 30], [10, 30]],
            "confidence": 0.95,
        },
        {
            "text": "john.smith@example.com",
            "bbox": [[10, 40], [90, 40], [90, 50], [10, 50]],
            "confidence": 0.92,
        },
        {
            "text": "+90 532 123 4567",
            "bbox": [[10, 60], [90, 60], [90, 70], [10, 70]],
            "confidence": 0.88,
        },
        {
            "text": "TR12 3456 7890 1234 5678 9012 34",
            "bbox": [[10, 80], [90, 80], [90, 90], [10, 90]],
            "confidence": 0.91,
        },
    ]


@pytest.fixture
def sample_detection_result() -> list[dict[str, Any]]:
    """Sample field detection result."""
    return [
        {
            "text": "John Smith",
            "field_type": "PERSON_NAME",
            "confidence": 0.95,
            "bbox": [[10, 20], [90, 20], [90, 30], [10, 30]],
            "risk_level": "HIGH",
            "regulations": ["GDPR", "KVKK"],
        },
        {
            "text": "john.smith@example.com",
            "field_type": "EMAIL",
            "confidence": 0.92,
            "bbox": [[10, 40], [90, 40], [90, 50], [10, 50]],
            "risk_level": "HIGH",
            "regulations": ["GDPR", "KVKK", "CCPA"],
        },
    ]


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Mock LLM client for testing without actual API calls."""
    client = AsyncMock()
    client.post = AsyncMock(
        return_value=MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={"choices": [{"message": {"content": '{"sensitive_fields": []}'}}]}
            ),
        )
    )
    return client


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "anonymization": {
            "secret_key": "test_secret_key_for_testing",
            "persist_registry": False,
        },
        "ocr_settings": {
            "dpi": 300,
            "preprocessing": {
                "enabled": False,
            },
        },
        "detection_rules": {
            "use_llm_classification": False,
            "use_fallback_detection": True,
            "min_confidence": 0.60,
            "auto_mask_confidence": 0.85,
        },
        "masking_strategy": {
            "text_fields": {
                "method": "white_box_overlay",
                "background_color": [255, 255, 255],
                "padding": 5,
            },
        },
        "verification": {
            "enabled": False,
        },
    }


# Test data constants
TEST_NAMES = [
    "John Smith",
    "Jane Doe",
    "Mehmet Yilmaz",
    "Hans Mueller",
    "Marie Dupont",
]

TEST_EMAILS = [
    "john@example.com",
    "test.user@company.co.uk",
    "info@firma.com.tr",
]

TEST_PHONES = [
    "+90 532 123 4567",
    "+1 555 123 4567",
    "+49 170 1234567",
]

TEST_COMPANIES = [
    "Acme Corporation Ltd",
    "Tech Solutions GmbH",
    "Yazilim A.S.",
    "Global Services Inc",
]
