"""
Anonymization Engine Module

Deterministic anonymization for document anonymization.
Uses realistic dummy data generation with cross-document consistency.
"""

import hashlib
import hmac
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .constants import EntityNamespace, get_namespace_for_field
from .dummy_generator import DummyDataGenerator

logger = logging.getLogger(__name__)


@dataclass
class TokenRecord:
    """
    Record for a generated token.

    Stores metadata about the token without storing original value.
    GDPR/Privacy compliant: original value is NOT stored.
    """

    token: str
    namespace: str
    normalized_hash: str  # Hash of normalized original (not reversible)
    created_at: str
    document_id: Optional[str] = None
    field_context: Optional[str] = None
    confidence: float = 1.0


@dataclass
class TokenRegistry:
    """
    Registry of generated tokens.

    Maintains consistency across documents by mapping
    normalized text hashes to tokens.
    """

    tokens: Dict[str, TokenRecord] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"

    def to_dict(self) -> Dict:
        """Convert registry to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "token_count": len(self.tokens),
            "tokens": {
                k: {
                    "token": v.token,
                    "namespace": v.namespace,
                    "normalized_hash": v.normalized_hash,
                    "created_at": v.created_at,
                    "document_id": v.document_id,
                    "field_context": v.field_context,
                    "confidence": v.confidence,
                }
                for k, v in self.tokens.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TokenRegistry":
        """Create registry from dictionary."""
        registry = cls(
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            version=data.get("version", "1.0"),
        )
        for k, v in data.get("tokens", {}).items():
            registry.tokens[k] = TokenRecord(
                token=v["token"],
                namespace=v["namespace"],
                normalized_hash=v["normalized_hash"],
                created_at=v["created_at"],
                document_id=v.get("document_id"),
                field_context=v.get("field_context"),
                confidence=v.get("confidence", 1.0),
            )
        return registry


class AnonymizationEngine:
    """
    HMAC-SHA256 based deterministic tokenization engine.

    Features:
    - Deterministic: same input always produces same output
    - Cross-document consistent: entities get same token everywhere
    - Cryptographically secure: not reversible without secret key
    - Privacy compliant: original values not stored
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        registry_path: Optional[str] = None,
        persist_registry: bool = True,
        use_realistic_dummy: bool = True,
        locale: str = "tr_TR",
    ):
        """
        Initialize anonymization engine.

        Args:
            secret_key: HMAC secret key (uses env var if not provided)
            registry_path: Path to persist token registry
            persist_registry: Whether to auto-save registry
            use_realistic_dummy: Use Faker-based realistic data instead of tokens
            locale: Locale for dummy data generation
        """
        # Get secret key from env if not provided
        self.secret_key = secret_key or os.getenv(
            "ANONYMIZATION_SECRET_KEY", "document_anonymizer_default_key_change_in_production"
        )

        # Registry for cross-document consistency
        self.registry = TokenRegistry()
        self.registry_path = registry_path
        self.persist_registry = persist_registry

        # Load existing registry if available
        if registry_path and Path(registry_path).exists():
            self._load_registry(registry_path)

        # Cache for performance
        self._token_cache: Dict[str, str] = {}

        # Dummy data generator for realistic anonymization
        self.use_realistic_dummy = use_realistic_dummy
        self._dummy_generator = (
            DummyDataGenerator(secret_key=self.secret_key, locale=locale)
            if use_realistic_dummy
            else None
        )

        logger.debug(f"AnonymizationEngine initialized (realistic_dummy={use_realistic_dummy})")

    def anonymize(
        self,
        original_text: str,
        field_type: str,
        document_id: Optional[str] = None,
        context: Optional[str] = None,
        confidence: float = 1.0,
    ) -> str:
        """
        Anonymize text by generating deterministic replacement.

        Args:
            original_text: Original sensitive text
            field_type: Type of field (for dummy data selection)
            document_id: Optional document identifier for audit
            context: Optional context information
            confidence: Detection confidence

        Returns:
            Anonymized text (realistic dummy or token)
        """
        if not original_text or not original_text.strip():
            return original_text

        # Normalize text for consistent matching (field-type aware)
        normalized = self._normalize_text(original_text, field_type)

        # Check cache first - use only normalized text for consistency
        cache_key = normalized
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        # Try to find similar existing value (post-processing consistency)
        similar_match = self._find_similar_cached_value(normalized, field_type)
        if similar_match:
            logger.debug("Found similar match, using existing replacement")
            self._token_cache[cache_key] = similar_match
            return similar_match

        # Generate replacement
        if self.use_realistic_dummy and self._dummy_generator:
            # Use realistic dummy data
            replacement = self._dummy_generator.generate(
                original_text=original_text, field_type=field_type, context=context
            )
        else:
            # Fall back to token-based anonymization
            namespace = get_namespace_for_field(field_type)
            replacement = self._generate_token(normalized, namespace)

        # Create hash for registry (not reversible)
        normalized_hash = self._hash_text(normalized)

        # Store in registry
        if normalized_hash not in self.registry.tokens:
            self.registry.tokens[normalized_hash] = TokenRecord(
                token=replacement,
                namespace=field_type,
                normalized_hash=normalized_hash,
                created_at=datetime.utcnow().isoformat(),
                document_id=document_id,
                field_context=context,
                confidence=confidence,
            )

        # Cache for performance - keyed by normalized text only
        self._token_cache[cache_key] = replacement

        # Also store original normalized for similarity matching
        self._store_normalized_value(normalized, replacement, field_type)

        return replacement

    def _find_similar_cached_value(self, normalized: str, field_type: str) -> Optional[str]:
        """Find similar cached value for OCR consistency across variations."""
        if not hasattr(self, "_normalized_values"):
            return None

        # For names and IDs, check if this is a substring of existing value
        # or if existing value is substring of this
        for stored_normalized, (stored_replacement, stored_type) in self._normalized_values.items():
            # Skip if different field types (name vs number)
            if self._is_incompatible_type(field_type, stored_type):
                continue

            # Exact match already handled by cache
            if normalized == stored_normalized:
                return stored_replacement

            # Skip very short strings to avoid false matches
            if len(normalized) < 3 or len(stored_normalized) < 3:
                continue

            # Substring matching for longer values
            if len(normalized) >= 4 and len(stored_normalized) >= 4:
                # Check if one is substring of other (with some tolerance)
                if normalized in stored_normalized or stored_normalized in normalized:
                    return stored_replacement

                # Check word-level overlap for names
                if field_type in ("person_name", "company_name"):
                    overlap = self._calculate_word_overlap(normalized, stored_normalized)
                    if overlap >= 0.5:  # At least 50% word overlap
                        return stored_replacement

            # For numeric IDs, check if digits match
            if field_type in ("national_id", "tax_id", "phone"):
                norm_digits = re.sub(r"\D", "", normalized)
                stored_digits = re.sub(r"\D", "", stored_normalized)
                if norm_digits and stored_digits:
                    if norm_digits == stored_digits:
                        return stored_replacement
                    # Partial match for IDs (one contains the other)
                    if len(norm_digits) >= 8 and len(stored_digits) >= 8:
                        if norm_digits in stored_digits or stored_digits in norm_digits:
                            return stored_replacement

        return None

    def _store_normalized_value(self, normalized: str, replacement: str, field_type: str) -> None:
        """Store normalized value for similarity matching."""
        if not hasattr(self, "_normalized_values"):
            self._normalized_values: Dict[str, tuple] = {}

        self._normalized_values[normalized] = (replacement, field_type)

    def _is_incompatible_type(self, type1: str, type2: str) -> bool:
        """Check if two field types are incompatible for matching."""
        name_types = {"person_name", "company_name"}
        id_types = {"national_id", "tax_id", "passport"}
        contact_types = {"phone", "email"}

        type1_category = None
        type2_category = None

        for cat, types in [("name", name_types), ("id", id_types), ("contact", contact_types)]:
            if type1 in types:
                type1_category = cat
            if type2 in types:
                type2_category = cat

        # If both have categories and they're different, incompatible
        if type1_category and type2_category and type1_category != type2_category:
            return True

        return False

    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word-level overlap ratio between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _normalize_text(self, text: str, field_type: Optional[str] = None) -> str:
        """
        Normalize text for consistent matching.

        Handles OCR errors and variations:
        - Field-type specific normalization
        - Case normalization
        - Whitespace normalization
        - Unicode normalization
        - Common character substitutions
        - Label extraction

        Args:
            text: Original text
            field_type: Optional field type for specific normalization

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # First, extract value from labeled text
        extracted = self._extract_value_from_labeled_text(text)

        # Unicode normalization
        normalized = unicodedata.normalize("NFKC", extracted)

        # Field-type specific normalization
        field_type_lower = (field_type or "").lower()

        # License plates: remove all whitespace/dashes, uppercase
        if "plate" in field_type_lower or "license" in field_type_lower:
            normalized = re.sub(r"[\s\-]+", "", normalized).upper()
            return normalized.strip()

        # Phone numbers: keep only digits and + sign
        if "phone" in field_type_lower or "tel" in field_type_lower or "fax" in field_type_lower:
            normalized = re.sub(r"[^\d+]", "", normalized)
            return normalized

        # Email: lowercase, no extra normalization needed
        if "email" in field_type_lower:
            return normalized.lower().strip()

        # National IDs: remove all whitespace and dashes for matching
        if any(
            x in field_type_lower
            for x in ["ssn", "nino", "insee", "national_id", "tax_id", "id_number"]
        ):
            normalized = re.sub(r"[\s\-.]", "", normalized).upper()
            return normalized

        # Default normalization for other types
        # Lowercase
        normalized = normalized.lower()

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        # Common OCR error corrections
        ocr_corrections = {
            "#": "",  # OCR artifact
            "_": "",  # OCR artifact
            "ı": "i",  # Turkish ı
            "İ": "i",  # Turkish İ
            "ğ": "g",  # Turkish ğ
            "ü": "u",  # Turkish ü
            "ş": "s",  # Turkish ş
            "ö": "o",  # Turkish ö
            "ç": "c",  # Turkish ç
        }

        for old, new in ocr_corrections.items():
            normalized = normalized.replace(old, new)

        # Common OCR substitutions for letters
        letter_substitutions = {
            "0": "o",  # Zero vs O
            "1": "l",  # One vs L
            "|": "l",  # Pipe vs L
            "!": "l",  # Exclamation vs L
        }

        # Only apply to text that looks like a word (not numbers)
        if not normalized.replace(" ", "").isdigit():
            for old, new in letter_substitutions.items():
                normalized = normalized.replace(old, new)

        # Remove common punctuation for matching
        normalized = re.sub(r"[.,;:!?'\"-]", "", normalized)

        return normalized.strip()

    def _extract_value_from_labeled_text(self, text: str) -> str:
        """Extract value part from labeled text (e.g., 'ADI: OZKAN' → 'OZKAN')."""
        # Common label patterns to remove
        label_patterns = [
            r"^T\.?C\.?\s*K[Ii][Mm][Ll][Ii][Kk]\s*N[Oo]\.?\s*:?\s*",  # TC Kimlik No
            r"^K[Ii][Mm][Ll][Ii][Kk]\s*N[Oo]\.?\s*:?\s*",  # Kimlik No
            r"^AD[Ii]?\s*:?\s*",  # ADI
            r"^SOYAD[Ii]?\s*:?\s*",  # SOYADI
            r"^AD[Ii]?\s*SOYAD[Ii]?\s*:?\s*",  # ADI SOYADI
            r"^V\.?K\.?N\.?\s*:?\s*",  # VKN
            r"^PLAKA\s*:?\s*",  # PLAKA
            r"^TEL\.?\s*:?\s*",  # TEL
            r"^GSM\s*:?\s*",  # GSM
            r"^E-?POSTA\s*:?\s*",  # E-POSTA
            r"^ADRES\s*:?\s*",  # ADRES
        ]

        result = text.strip()
        for pattern in label_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        return result.strip() if result.strip() else text

    def _generate_token(self, normalized_text: str, namespace: EntityNamespace) -> str:
        """
        Generate HMAC-based token.

        Args:
            normalized_text: Normalized text
            namespace: Entity namespace

        Returns:
            Token string (e.g., "PER-A7B3C912")
        """
        # Create HMAC with namespace for uniqueness
        message = f"{namespace.value}:{normalized_text}"

        hash_value = (
            hmac.new(self.secret_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256)
            .hexdigest()[:8]
            .upper()
        )

        return f"{namespace.value}-{hash_value}"

    def _hash_text(self, text: str) -> str:
        """Create non-reversible hash of text for registry."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def update_locale(self, locale: str) -> None:
        """
        Update the locale for dummy data generation.

        Args:
            locale: Faker locale (e.g., 'tr_TR', 'de_DE', 'fr_FR')
        """
        if self._dummy_generator:
            self._dummy_generator.update_locale(locale)

    def save_registry(self, path: Optional[str] = None) -> None:
        """
        Save token registry to file.

        Args:
            path: Output path (uses default if not provided)
        """
        import json

        save_path = path or self.registry_path
        if not save_path:
            logger.warning("No registry path specified, skipping save")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.registry.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Token registry saved: {save_path} ({len(self.registry.tokens)} tokens)")

    def _load_registry(self, path: str) -> None:
        """Load token registry from file."""
        import json

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.registry = TokenRegistry.from_dict(data)
            logger.info(f"Token registry loaded: {path} ({len(self.registry.tokens)} tokens)")

        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")

    def get_statistics(self) -> Dict:
        """Get engine statistics."""
        namespace_counts: Dict[str, int] = {}
        for record in self.registry.tokens.values():
            ns = record.namespace
            namespace_counts[ns] = namespace_counts.get(ns, 0) + 1

        return {
            "total_tokens": len(self.registry.tokens),
            "by_namespace": namespace_counts,
            "cache_size": len(self._token_cache),
        }


def create_anonymization_engine(
    secret_key: Optional[str] = None, registry_path: Optional[str] = None
) -> AnonymizationEngine:
    """
    Create a new anonymization engine instance.

    Args:
        secret_key: Optional secret key
        registry_path: Optional registry path

    Returns:
        New AnonymizationEngine instance
    """
    return AnonymizationEngine(secret_key=secret_key, registry_path=registry_path)


# Singleton instance
_engine_instance: Optional[AnonymizationEngine] = None


def get_anonymization_engine(
    secret_key: Optional[str] = None, registry_path: Optional[str] = None
) -> AnonymizationEngine:
    """
    Get singleton anonymization engine instance.

    Args:
        secret_key: Optional secret key (only used on first call)
        registry_path: Optional registry path (only used on first call)

    Returns:
        Singleton AnonymizationEngine instance
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = create_anonymization_engine(
            secret_key=secret_key, registry_path=registry_path
        )
    return _engine_instance
