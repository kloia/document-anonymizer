"""Tests for anonymization_engine module."""

import json
import tempfile
from pathlib import Path

import pytest

from document_anonymizer.anonymization_engine import (
    AnonymizationEngine,
    TokenRecord,
    TokenRegistry,
    get_anonymization_engine,
)


class TestTokenRecord:
    """Tests for TokenRecord dataclass."""

    def test_creation(self):
        """TokenRecord should be created with required fields."""
        record = TokenRecord(
            token="Alex Johnson",
            namespace="PERSON_NAME",
            normalized_hash="abc123",
            created_at="2024-01-01T00:00:00",
        )
        assert record.token == "Alex Johnson"
        assert record.namespace == "PERSON_NAME"
        assert record.confidence == 1.0  # default

    def test_optional_fields(self):
        """TokenRecord should accept optional fields."""
        record = TokenRecord(
            token="test",
            namespace="TEST",
            normalized_hash="hash",
            created_at="2024-01-01",
            document_id="doc.pdf",
            field_context="page:1",
            confidence=0.95,
        )
        assert record.document_id == "doc.pdf"
        assert record.field_context == "page:1"
        assert record.confidence == 0.95


class TestTokenRegistry:
    """Tests for TokenRegistry dataclass."""

    def test_creation(self):
        """Registry should be created with defaults."""
        registry = TokenRegistry()
        assert registry.tokens == {}
        assert registry.version == "1.0"

    def test_to_dict(self):
        """Registry should convert to dictionary."""
        registry = TokenRegistry()
        registry.tokens["hash1"] = TokenRecord(
            token="test_token",
            namespace="TEST",
            normalized_hash="hash1",
            created_at="2024-01-01",
        )

        data = registry.to_dict()
        assert "version" in data
        assert "tokens" in data
        assert data["token_count"] == 1
        assert "hash1" in data["tokens"]

    def test_from_dict(self):
        """Registry should be recreated from dictionary."""
        original = TokenRegistry()
        original.tokens["hash1"] = TokenRecord(
            token="test_token",
            namespace="TEST",
            normalized_hash="hash1",
            created_at="2024-01-01",
            confidence=0.9,
        )

        data = original.to_dict()
        restored = TokenRegistry.from_dict(data)

        assert len(restored.tokens) == 1
        assert "hash1" in restored.tokens
        assert restored.tokens["hash1"].token == "test_token"
        assert restored.tokens["hash1"].confidence == 0.9


class TestAnonymizationEngine:
    """Tests for AnonymizationEngine class."""

    @pytest.fixture
    def engine(self):
        """Create engine instance for testing."""
        return AnonymizationEngine(
            secret_key="test_secret_key",
            persist_registry=False,
            use_realistic_dummy=True,
        )

    def test_initialization(self, engine):
        """Engine should initialize with correct settings."""
        assert engine.secret_key == "test_secret_key"
        assert engine.use_realistic_dummy is True
        assert len(engine.registry.tokens) == 0

    def test_anonymize_empty_string(self, engine):
        """Empty string should return unchanged."""
        assert engine.anonymize("", "person_name") == ""
        assert engine.anonymize("   ", "person_name") == "   "

    def test_anonymize_consistency(self, engine):
        """Same input should always produce same output."""
        result1 = engine.anonymize("John Smith", "person_name")
        result2 = engine.anonymize("John Smith", "person_name")
        assert result1 == result2

    def test_anonymize_different_inputs(self, engine):
        """Different inputs should produce different outputs."""
        result1 = engine.anonymize("John Smith", "person_name")
        result2 = engine.anonymize("Jane Doe", "person_name")
        assert result1 != result2

    def test_anonymize_updates_registry(self, engine):
        """Anonymization should add entry to registry."""
        engine.anonymize("Test Person", "person_name")
        assert len(engine.registry.tokens) == 1

    def test_anonymize_with_context(self, engine):
        """Anonymization should store context in registry."""
        engine.anonymize(
            "Test Person",
            "person_name",
            document_id="test.pdf",
            context="page:1",
            confidence=0.95,
        )
        # Find the token in registry
        for record in engine.registry.tokens.values():
            if record.namespace == "person_name":
                assert record.document_id == "test.pdf"
                assert record.field_context == "page:1"
                assert record.confidence == 0.95

    def test_caching(self, engine):
        """Results should be cached for performance."""
        engine.anonymize("Test Value", "person_name")
        assert len(engine._token_cache) == 1

    def test_statistics(self, engine):
        """Statistics should track engine state."""
        engine.anonymize("Person 1", "person_name")
        engine.anonymize("Person 2", "person_name")
        engine.anonymize("test@example.com", "email")

        stats = engine.get_statistics()
        assert stats["total_tokens"] == 3
        assert stats["cache_size"] == 3
        assert "by_namespace" in stats


class TestAnonymizationEngineNormalization:
    """Tests for text normalization in AnonymizationEngine."""

    @pytest.fixture
    def engine(self):
        return AnonymizationEngine(
            secret_key="test_key",
            persist_registry=False,
        )

    def test_case_normalization(self, engine):
        """Different cases should produce same result."""
        result1 = engine.anonymize("JOHN SMITH", "person_name")
        result2 = engine.anonymize("john smith", "person_name")
        assert result1 == result2

    def test_whitespace_normalization(self, engine):
        """Extra whitespace should be normalized."""
        result1 = engine.anonymize("John  Smith", "person_name")
        result2 = engine.anonymize("John Smith", "person_name")
        assert result1 == result2

    def test_normalize_text_empty(self, engine):
        """Empty text should normalize to empty string."""
        assert engine._normalize_text("") == ""
        assert engine._normalize_text(None) == ""

    def test_normalize_removes_punctuation(self, engine):
        """Punctuation should be removed for matching."""
        normalized = engine._normalize_text("John, Smith!")
        assert "," not in normalized
        assert "!" not in normalized


class TestAnonymizationEngineTokenMode:
    """Tests for token-based anonymization mode."""

    @pytest.fixture
    def engine(self):
        """Create engine without dummy data."""
        return AnonymizationEngine(
            secret_key="test_key",
            persist_registry=False,
            use_realistic_dummy=False,
        )

    def test_generates_token_format(self, engine):
        """Token mode should generate PREFIX-HASH format."""
        result = engine.anonymize("John Smith", "PERSON_NAME")
        # Token format: PREFIX-XXXXXXXX
        assert "-" in result
        parts = result.split("-")
        assert len(parts) == 2

    def test_token_consistency(self, engine):
        """Same input should produce same token."""
        result1 = engine.anonymize("Test Input", "PERSON_NAME")
        result2 = engine.anonymize("Test Input", "PERSON_NAME")
        assert result1 == result2


class TestAnonymizationEngineRegistryPersistence:
    """Tests for registry save/load functionality."""

    def test_save_registry(self):
        """Registry should be saveable to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"
            engine = AnonymizationEngine(
                secret_key="test_key",
                registry_path=str(registry_path),
                persist_registry=True,
            )

            engine.anonymize("Test Person", "person_name")
            engine.save_registry()

            assert registry_path.exists()
            with open(registry_path) as f:
                data = json.load(f)
                assert "tokens" in data
                assert data["token_count"] == 1

    def test_load_registry(self):
        """Registry should be loadable from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"

            # Create initial registry
            engine1 = AnonymizationEngine(
                secret_key="test_key",
                registry_path=str(registry_path),
            )
            engine1.anonymize("Test Person", "person_name")
            engine1.save_registry()

            # Load in new engine
            engine2 = AnonymizationEngine(
                secret_key="test_key",
                registry_path=str(registry_path),
            )

            assert len(engine2.registry.tokens) == 1

    def test_save_creates_directory(self):
        """save_registry should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "nested" / "dir" / "registry.json"
            engine = AnonymizationEngine(
                secret_key="test_key",
                registry_path=str(registry_path),
            )

            engine.anonymize("Test", "person_name")
            engine.save_registry()

            assert registry_path.exists()

    def test_save_without_path_logs_warning(self):
        """save_registry without path should not raise."""
        engine = AnonymizationEngine(
            secret_key="test_key",
            registry_path=None,
        )
        # Should not raise
        engine.save_registry()


class TestGetAnonymizationEngine:
    """Tests for singleton factory function."""

    def test_returns_engine(self):
        """Should return AnonymizationEngine instance."""
        engine = get_anonymization_engine(secret_key="test_key")
        assert isinstance(engine, AnonymizationEngine)

    def test_singleton_pattern(self):
        """Should return same instance on multiple calls."""
        # Note: This test may be affected by other tests due to global state
        # In production, you might want to reset the global instance
        engine1 = get_anonymization_engine()
        engine2 = get_anonymization_engine()
        assert engine1 is engine2


class TestCrossDocumentConsistency:
    """Tests for cross-document token consistency."""

    def test_same_value_same_token(self):
        """Same value in different contexts should get same token."""
        engine = AnonymizationEngine(
            secret_key="cross_doc_test",
            persist_registry=False,
        )

        # Simulate different documents
        result1 = engine.anonymize(
            "John Smith",
            "person_name",
            document_id="doc1.pdf",
        )
        result2 = engine.anonymize(
            "John Smith",
            "person_name",
            document_id="doc2.pdf",
        )

        assert result1 == result2

    def test_registry_tracks_first_occurrence(self):
        """Registry should track where value first appeared."""
        engine = AnonymizationEngine(
            secret_key="track_test",
            persist_registry=False,
        )

        engine.anonymize(
            "Unique Person",
            "person_name",
            document_id="first_doc.pdf",
        )

        # Second occurrence with different document
        engine.anonymize(
            "Unique Person",
            "person_name",
            document_id="second_doc.pdf",
        )

        # Should still have only one registry entry
        assert len(engine.registry.tokens) == 1

        # First document should be recorded
        record = list(engine.registry.tokens.values())[0]
        assert record.document_id == "first_doc.pdf"
