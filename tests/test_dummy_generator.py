"""Tests for dummy_generator module."""

import pytest

from document_anonymizer.dummy_generator import (
    DummyDataGenerator,
    analyze_pattern,
    generate_from_pattern,
)


class TestAnalyzePattern:
    """Tests for analyze_pattern function."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert analyze_pattern("") == []

    def test_digits_only(self):
        """Pure digits should return single D segment."""
        result = analyze_pattern("12345")
        assert result == [("D", 5)]

    def test_letters_only(self):
        """Pure letters should return single L segment."""
        result = analyze_pattern("ABC")
        assert result == [("L", 3)]

    def test_license_plate_format(self):
        """Turkish license plate format should be correctly parsed."""
        result = analyze_pattern("34 KLY 482")
        assert result == [("D", 2), ("S", 1), ("L", 3), ("S", 1), ("D", 3)]

    def test_phone_format(self):
        """Phone number format should preserve separators."""
        result = analyze_pattern("+90 532 123 4567")
        # + is separator, then digits, space, digits, space, digits, space, digits
        assert ("D", 2) in result  # 90
        assert ("D", 3) in result  # 532, 123
        assert ("D", 4) in result  # 4567

    def test_letters_digits_format(self):
        """Mixed letters and digits should be parsed correctly."""
        result = analyze_pattern("AB12")
        assert result == [("L", 2), ("D", 2)]

    def test_mixed_separators(self):
        """Mixed separators should be handled."""
        result = analyze_pattern("12-34/56")
        assert ("S", 1) in result


class TestGenerateFromPattern:
    """Tests for generate_from_pattern function."""

    def test_preserves_length(self):
        """Generated text should match original length."""
        original = "34 KLY 482"
        pattern = analyze_pattern(original)
        result = generate_from_pattern(pattern, original, seed=12345)
        assert len(result) == len(original)

    def test_preserves_separators(self):
        """Separators should be preserved exactly."""
        original = "12-34/56"
        pattern = analyze_pattern(original)
        result = generate_from_pattern(pattern, original, seed=12345)
        assert "-" in result
        assert "/" in result

    def test_deterministic(self):
        """Same seed should produce same result."""
        original = "ABC 123"
        pattern = analyze_pattern(original)
        result1 = generate_from_pattern(pattern, original, seed=42)
        result2 = generate_from_pattern(pattern, original, seed=42)
        assert result1 == result2

    def test_different_seeds(self):
        """Different seeds should produce different results."""
        original = "ABC 123"
        pattern = analyze_pattern(original)
        result1 = generate_from_pattern(pattern, original, seed=42)
        result2 = generate_from_pattern(pattern, original, seed=99)
        assert result1 != result2


class TestDummyDataGenerator:
    """Tests for DummyDataGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return DummyDataGenerator(secret_key="test_key", locale="en_US")

    def test_initialization(self, generator):
        """Generator should initialize correctly."""
        assert generator.secret_key == "test_key"
        assert generator.locale == "en_US"

    def test_empty_input(self, generator):
        """Empty input should return empty output."""
        assert generator.generate("", "person_name") == ""
        assert generator.generate("   ", "person_name") == "   "

    def test_consistency(self, generator):
        """Same input should always produce same output."""
        result1 = generator.generate("John Smith", "person_name")
        result2 = generator.generate("John Smith", "person_name")
        assert result1 == result2

    def test_caching(self, generator):
        """Results should be cached."""
        generator.generate("Test Value", "person_name")
        assert "Test Value:person_name" in generator._cache


class TestNameGeneration:
    """Tests for name generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_single_name(self, generator):
        """Single name should produce single word."""
        result = generator.generate("John", "person_name")
        assert " " not in result
        assert len(result) > 0

    def test_two_part_name(self, generator):
        """Two-part name should produce two words."""
        result = generator.generate("John Smith", "person_name")
        assert " " in result
        parts = result.split()
        assert len(parts) == 2

    def test_three_part_name(self, generator):
        """Three-part name should produce three words."""
        result = generator.generate("John Middle Smith", "person_name")
        parts = result.split()
        assert len(parts) == 3


class TestCompanyGeneration:
    """Tests for company name generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_preserves_ltd(self, generator):
        """Should preserve Ltd suffix."""
        result = generator.generate("Acme Corporation Ltd", "company_name")
        assert result.endswith("Ltd")

    def test_preserves_gmbh(self, generator):
        """Should preserve GmbH suffix."""
        result = generator.generate("Tech Solutions GmbH", "company_name")
        assert result.endswith("GmbH")

    def test_preserves_as(self, generator):
        """Should preserve A.S. suffix."""
        result = generator.generate("Yazilim A.S.", "company_name")
        assert result.endswith("A.S.")

    def test_no_suffix(self, generator):
        """Company without suffix should work."""
        result = generator.generate("Simple Company", "company_name")
        assert len(result) > 0


class TestEmailGeneration:
    """Tests for email generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_preserves_at_symbol(self, generator):
        """Email should contain @ symbol."""
        result = generator.generate("john@example.com", "email")
        assert "@" in result

    def test_preserves_tld(self, generator):
        """Email should preserve TLD structure."""
        result = generator.generate("user@company.co.uk", "email")
        assert result.endswith(".uk")

    def test_turkish_tld(self, generator):
        """Should preserve .tr TLD."""
        result = generator.generate("info@firma.com.tr", "email")
        assert result.endswith(".tr")

    def test_invalid_email(self, generator):
        """Invalid email should return generic format."""
        result = generator.generate("not-an-email", "email")
        assert "@" in result


class TestPhoneGeneration:
    """Tests for phone number generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_preserves_format(self, generator):
        """Phone format should be preserved."""
        original = "+90 532 123 4567"
        result = generator.generate(original, "phone")
        # Should have same number of spaces
        assert result.count(" ") == original.count(" ")

    def test_preserves_length(self, generator):
        """Phone length should be preserved."""
        original = "0532 123 45 67"
        result = generator.generate(original, "phone")
        assert len(result) == len(original)


class TestLicensePlateGeneration:
    """Tests for license plate generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_turkish_format(self, generator):
        """Turkish plate format should be preserved."""
        original = "34 KLY 482"
        result = generator.generate(original, "license_plate")
        # Should have same structure: DD LLL DDD
        pattern = analyze_pattern(result)
        expected_pattern = analyze_pattern(original)
        assert pattern == expected_pattern

    def test_german_format(self, generator):
        """German plate format should be preserved."""
        original = "M-AB 1234"
        result = generator.generate(original, "license_plate")
        # Should preserve dash and space positions
        assert "-" in result
        assert " " in result


class TestNationalIDGeneration:
    """Tests for national ID generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_preserves_length(self, generator):
        """National ID length should be preserved."""
        original = "12345678901"  # 11 digits like Turkish TC
        result = generator.generate(original, "national_id")
        assert len(result) == len(original)

    def test_formatted_id(self, generator):
        """Formatted ID should preserve format."""
        original = "123-45-6789"  # US SSN format
        result = generator.generate(original, "national_id")
        assert result.count("-") == original.count("-")


class TestSignatureAndStampPlaceholders:
    """Tests for signature and stamp placeholders."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_signature_placeholder(self, generator):
        """Signature should return placeholder."""
        result = generator.generate("any text", "signature")
        assert result == "[SIGNATURE]"

    def test_stamp_placeholder(self, generator):
        """Stamp should return placeholder."""
        result = generator.generate("any text", "stamp")
        assert result == "[STAMP]"


class TestGenericGeneration:
    """Tests for generic/unknown field type generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_unknown_type(self, generator):
        """Unknown field type should use generic generator."""
        result = generator.generate("TEST123", "unknown_type")
        assert len(result) == 7

    def test_digits_only(self, generator):
        """Digit-only input should produce digits."""
        result = generator.generate("12345", "unknown_type")
        assert result.isdigit()

    def test_letters_only(self, generator):
        """Letter-only input should produce letters."""
        result = generator.generate("ABCDE", "unknown_type")
        assert result.isalpha()


class TestUpdateLocale:
    """Tests for locale update functionality."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key", locale="en_US")

    def test_update_locale_changes_locale(self, generator):
        """Should update locale when different."""
        assert generator.locale == "en_US"
        generator.update_locale("tr_TR")
        assert generator.locale == "tr_TR"

    def test_update_locale_same_locale(self, generator):
        """Should not change when same locale."""
        generator.update_locale("en_US")
        assert generator.locale == "en_US"


class TestAnalyzePatternMixed:
    """Tests for pattern analysis with special characters."""

    def test_special_characters(self):
        """Should handle special characters as 'X' type."""
        result = analyze_pattern("test@#$test")
        # @ is separator, # and $ are mixed
        types = [t for t, _ in result]
        assert "X" in types or "S" in types  # @ counts as separator

    def test_mixed_alphanumeric(self):
        """Should handle mixed patterns."""
        result = analyze_pattern("A1B2C3")
        assert len(result) == 6  # Alternating


class TestDateGeneration:
    """Tests for date generation with different formats."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_slash_format(self, generator):
        """Should preserve slash date format."""
        result = generator.generate("15/06/1990", "date")
        assert "/" in result

    def test_dot_format(self, generator):
        """Should preserve dot date format."""
        result = generator.generate("15.06.1990", "date")
        assert "." in result

    def test_dash_format(self, generator):
        """Should preserve ISO date format."""
        result = generator.generate("1990-06-15", "date")
        assert "-" in result

    def test_no_separator(self, generator):
        """Should use default format when no separator."""
        result = generator.generate("19900615", "date")
        assert "." in result  # Default format


class TestAddressGeneration:
    """Tests for address generation with different structures."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_address_with_number(self, generator):
        """Should include street number."""
        result = generator.generate("123 Main Street", "address")
        # Should contain a number
        import re

        assert re.search(r"\d", result)

    def test_address_multiline(self, generator):
        """Should handle multi-line addresses."""
        result = generator.generate("123 Main St, City, Country", "address")
        assert "," in result

    def test_address_no_number(self, generator):
        """Should work without street number."""
        result = generator.generate("Main Street", "address")
        assert len(result) > 0


class TestTaxIdGeneration:
    """Tests for tax ID generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_tax_id_formatted(self, generator):
        """Should preserve formatted tax ID."""
        original = "12-3456789"
        result = generator.generate(original, "tax_id")
        assert "-" in result

    def test_tax_id_preserves_length(self, generator):
        """Should preserve tax ID length."""
        result = generator.generate("1234567890", "tax_id")
        assert len(result) == 10


class TestPassportGeneration:
    """Tests for passport number generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_passport_format_preserved(self, generator):
        """Should preserve passport format."""
        original = "AB 123456"
        result = generator.generate(original, "passport")
        assert " " in result

    def test_passport_short_input(self, generator):
        """Should handle short passport numbers."""
        result = generator.generate("A1", "passport")
        assert len(result) >= 2


class TestReferenceGeneration:
    """Tests for reference number generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_reference_preserves_length(self, generator):
        """Should match original reference length."""
        original = "REF-12345678"
        result = generator.generate(original, "reference")
        # Should match length excluding separators
        assert len(result) >= 8

    def test_reference_short_input(self, generator):
        """Should use minimum length for short input."""
        result = generator.generate("AB", "reference")
        assert len(result) >= 8


class TestInvoiceGeneration:
    """Tests for invoice number generation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_invoice_format_preserved(self, generator):
        """Should preserve invoice format."""
        original = "INV-2024-001"
        result = generator.generate(original, "invoice_number")
        assert "-" in result

    def test_invoice_preserves_length(self, generator):
        """Should preserve invoice length."""
        original = "INV123456"
        result = generator.generate(original, "invoice_number")
        assert len(result) == len(original)


class TestGenerateFromPatternMixed:
    """Tests for generate_from_pattern with edge cases."""

    def test_mixed_segment_type(self):
        """Should handle X (mixed) segment type."""
        pattern = [("X", 3)]
        result = generate_from_pattern(pattern, "ab@", seed=42)
        assert len(result) == 3

    def test_lowercase_preservation(self):
        """Should preserve lowercase in letters."""
        original = "abc"
        pattern = analyze_pattern(original)
        result = generate_from_pattern(pattern, original, seed=42)
        # Should be all lowercase
        assert result.islower()


class TestPhonePreservesFormat:
    """Tests for phone format preservation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_phone_preserves_length(self, generator):
        """Should preserve phone length."""
        original = "1234567890"
        result = generator.generate(original, "phone")
        assert len(result) == len(original)

    def test_phone_all_digits(self, generator):
        """Should produce digits for digit-only input."""
        result = generator.generate("5551234567", "phone")
        assert result.isdigit()


class TestNationalIdPreservesFormat:
    """Tests for national ID format preservation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_national_id_preserves_length(self, generator):
        """Should preserve length."""
        result = generator.generate("12345678901", "national_id")
        assert len(result) == 11

    def test_national_id_all_digits(self, generator):
        """Should produce digits for digit-only input."""
        result = generator.generate("12345678901", "national_id")
        assert result.isdigit()


class TestLicensePlatePreservesFormat:
    """Tests for license plate format preservation."""

    @pytest.fixture
    def generator(self):
        return DummyDataGenerator(secret_key="test_key")

    def test_plate_preserves_structure(self, generator):
        """Should preserve plate structure."""
        original = "ABC1234"
        result = generator.generate(original, "license_plate")
        assert len(result) == len(original)
