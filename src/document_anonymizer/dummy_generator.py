"""Format-preserving dummy data generator with seed-based consistency."""

import hashlib
import logging
import os
import random
import re
import string
from typing import Dict, List, Optional, Tuple

from faker import Faker

logger = logging.getLogger(__name__)


def analyze_pattern(text: str) -> List[Tuple[str, int]]:
    """Analyze text into (type, length) segments: D=digits, L=letters, S=separators, X=other."""
    if not text:
        return []

    segments = []
    current_type = None
    current_len = 0

    for char in text:
        if char.isdigit():
            char_type = "D"
        elif char.isalpha():
            char_type = "L"
        elif char in " -_./:()+":
            char_type = "S"  # Treat as separator (preserved as-is)
        else:
            char_type = "X"

        if char_type == current_type:
            current_len += 1
        else:
            if current_type is not None:
                segments.append((current_type, current_len))
            current_type = char_type
            current_len = 1

    if current_type is not None:
        segments.append((current_type, current_len))

    return segments


def generate_from_pattern(pattern: List[Tuple[str, int]], original: str, seed: int) -> str:
    """
    Generate text matching the given pattern.

    Args:
        pattern: List of (type, length) tuples
        original: Original text (for preserving separators)
        seed: Random seed

    Returns:
        Generated text matching pattern
    """
    random.seed(seed)
    result = []
    orig_idx = 0

    for seg_type, seg_len in pattern:
        if seg_type == "D":
            # Generate digits
            result.append("".join([str(random.randint(0, 9)) for _ in range(seg_len)]))
            orig_idx += seg_len
        elif seg_type == "L":
            # Generate letters (preserve case from original)
            orig_segment = (
                original[orig_idx : orig_idx + seg_len] if orig_idx < len(original) else ""
            )
            letters = []
            for i in range(seg_len):
                letter = random.choice(string.ascii_uppercase)
                # Preserve lowercase if original was lowercase
                if i < len(orig_segment) and orig_segment[i].islower():
                    letter = letter.lower()
                letters.append(letter)
            result.append("".join(letters))
            orig_idx += seg_len
        elif seg_type == "S":
            # Preserve original separator
            sep = (
                original[orig_idx : orig_idx + seg_len]
                if orig_idx < len(original)
                else " " * seg_len
            )
            result.append(sep)
            orig_idx += seg_len
        else:
            # Mixed - generate alphanumeric
            result.append(
                "".join(random.choices(string.ascii_uppercase + string.digits, k=seg_len))
            )
            orig_idx += seg_len

    return "".join(result)


class DummyDataGenerator:
    """Format-preserving dummy data generator. Same input always produces same output."""

    # Common name components for generating neutral names
    FIRST_NAMES = [
        "Alex",
        "Sam",
        "Jordan",
        "Taylor",
        "Morgan",
        "Casey",
        "Riley",
        "Quinn",
        "Avery",
        "Parker",
        "Cameron",
        "Drew",
        "Blake",
        "Logan",
        "Ryan",
        "Jamie",
    ]
    LAST_NAMES = [
        "Smith",
        "Johnson",
        "Brown",
        "Davis",
        "Wilson",
        "Moore",
        "Taylor",
        "Anderson",
        "Thomas",
        "Jackson",
        "White",
        "Harris",
        "Martin",
        "Garcia",
        "Miller",
        "Jones",
    ]

    # Common legal suffixes for companies (international)
    LEGAL_SUFFIXES = [
        "Ltd",
        "Ltd.",
        "LLC",
        "Inc",
        "Inc.",
        "Corp",
        "Corp.",
        "GmbH",
        "AG",
        "A.Ş.",
        "A.S.",
        "Ş.T.İ.",
        "S.A.",
        "S.L.",
        "B.V.",
        "N.V.",
        "Pty",
        "PLC",
        "LLP",
        "Co.",
        "Co",
        "& Co",
        "& Co.",
        "Limited",
        "Corporation",
    ]

    def __init__(self, secret_key: Optional[str] = None, locale: str = "en_US"):
        """Initialize generator with optional secret key and locale."""
        self.secret_key = secret_key or os.getenv(
            "ANONYMIZATION_SECRET_KEY", "document_anonymizer_default_key"
        )
        self.locale = locale

        # Cache for consistency
        self._cache: Dict[str, str] = {}

        # Field type to generator mapping
        self._generators = {
            "person_name": self._generate_name,
            "full_name": self._generate_name,
            "name": self._generate_name,
            "company_name": self._generate_company,
            "organization": self._generate_company,
            "phone": self._generate_phone,
            "email": self._generate_email,
            "address": self._generate_address,
            "street_address": self._generate_address,
            "tax_id": self._generate_tax_id,
            "national_id": self._generate_national_id,
            "passport": self._generate_passport,
            "passport_number": self._generate_passport,
            "license_plate": self._generate_license_plate,
            "vehicle_plate": self._generate_license_plate,
            "date_of_birth": self._generate_date,
            "dob": self._generate_date,
            "date": self._generate_date,
            "reference": self._generate_reference,
            "reference_number": self._generate_reference,
            "invoice_number": self._generate_invoice,
            "signature": self._generate_signature_placeholder,
            "stamp": self._generate_stamp_placeholder,
        }

        logger.debug(f"DummyDataGenerator initialized (locale: {locale})")

    def update_locale(self, locale: str) -> None:
        """
        Update the locale for dummy data generation.

        Args:
            locale: Faker locale (e.g., 'tr_TR', 'de_DE', 'fr_FR')
        """
        if locale != self.locale:
            logger.info(f"Updating DummyDataGenerator locale: {self.locale} → {locale}")
            self.locale = locale

    def generate(self, original_text: str, field_type: str, context: Optional[str] = None) -> str:
        """
        Generate dummy data for a field.

        Args:
            original_text: Original sensitive text
            field_type: Type of field
            context: Optional context for generation

        Returns:
            Realistic dummy data
        """
        if not original_text or not original_text.strip():
            return original_text

        # Create cache key
        cache_key = f"{original_text}:{field_type}"

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate seed from original text
        seed = self._generate_seed(original_text)

        # Get generator for field type
        generator = self._generators.get(field_type.lower(), self._generate_generic)

        # Generate dummy
        dummy = generator(seed, original_text)

        # Cache result
        self._cache[cache_key] = dummy

        return dummy

    def _generate_seed(self, text: str) -> int:
        """Generate consistent seed from text."""
        message = f"{self.secret_key}:{text.lower().strip()}"
        hash_hex = hashlib.sha256(message.encode("utf-8")).hexdigest()[:8]
        return int(hash_hex, 16)

    def _get_faker(self, seed: int) -> Faker:
        """Get seeded Faker instance."""
        fake = Faker(self.locale)
        fake.seed_instance(seed)
        return fake

    def _generate_name(self, seed: int, original: str) -> str:
        """Generate name preserving word count and approximate length."""
        random.seed(seed)

        # Count words in original
        words = original.split()
        word_count = len(words)

        if word_count == 0:
            return original

        # Generate matching number of name parts
        if word_count == 1:
            # Single name - could be first or last
            return random.choice(self.FIRST_NAMES + self.LAST_NAMES)
        elif word_count == 2:
            # First + Last
            return f"{random.choice(self.FIRST_NAMES)} {random.choice(self.LAST_NAMES)}"
        else:
            # Multiple parts - first + middle(s) + last
            parts = [random.choice(self.FIRST_NAMES)]
            for _ in range(word_count - 2):
                parts.append(random.choice(self.FIRST_NAMES))
            parts.append(random.choice(self.LAST_NAMES))
            return " ".join(parts)

    def _generate_company(self, seed: int, original: str) -> str:
        """Generate company name preserving legal suffix."""
        random.seed(seed)

        # Check for legal suffix
        found_suffix = None
        original_stripped = original.strip()

        for suffix in sorted(self.LEGAL_SUFFIXES, key=len, reverse=True):
            if original_stripped.endswith(suffix):
                found_suffix = suffix
                original_stripped = original_stripped[: -len(suffix)].strip()
                break

        # Generate company name parts
        company_words = [
            "Global",
            "United",
            "National",
            "Premier",
            "Alpha",
            "Delta",
            "Pacific",
            "Atlantic",
            "Northern",
            "Southern",
            "Central",
            "Metro",
            "Capital",
            "Crown",
            "Royal",
            "Summit",
            "Peak",
        ]
        company_types = [
            "Industries",
            "Solutions",
            "Services",
            "Systems",
            "Group",
            "Holdings",
            "Partners",
            "Associates",
            "Enterprises",
            "Trading",
        ]

        # Match approximate word count
        orig_words = original_stripped.split()
        if len(orig_words) <= 1:
            name = random.choice(company_words)
        else:
            name = f"{random.choice(company_words)} {random.choice(company_types)}"

        # Add back the legal suffix if present
        if found_suffix:
            return f"{name} {found_suffix}"
        return name

    def _generate_phone(self, seed: int, original: str) -> str:
        """Generate phone number preserving format."""
        random.seed(seed)

        # Analyze pattern of original
        pattern = analyze_pattern(original)

        if pattern:
            # Format-preserving generation
            return generate_from_pattern(pattern, original, seed)
        else:
            # Fallback: match length
            clean = re.sub(r"[^0-9]", "", original)
            length = len(clean) if len(clean) >= 7 else 10
            return "".join([str(random.randint(0, 9)) for _ in range(length)])

    def _generate_email(self, seed: int, original: str) -> str:
        """Generate email address preserving domain structure."""
        random.seed(seed)

        # Parse original email
        if "@" not in original:
            # Not a valid email, return generic
            return f"user{random.randint(100, 999)}@example.com"

        local_part, domain = original.rsplit("@", 1)

        # Generate new local part matching length
        new_local_length = max(len(local_part), 5)
        chars = string.ascii_lowercase + string.digits
        new_local = "".join(random.choices(chars, k=new_local_length))

        # Preserve domain structure but anonymize
        domain_parts = domain.split(".")
        if len(domain_parts) >= 2:
            # Keep TLD, anonymize rest
            tld = domain_parts[-1]
            new_domain = f"example.{tld}"
        else:
            new_domain = "example.com"

        return f"{new_local}@{new_domain}"

    def _generate_address(self, seed: int, original: str) -> str:
        """Generate address preserving structure."""
        random.seed(seed)

        # Generic address components
        streets = [
            "Main St",
            "Oak Ave",
            "Park Rd",
            "Cedar Ln",
            "Elm St",
            "Lake Dr",
            "Hill Rd",
            "River Rd",
        ]
        cities = ["Springfield", "Franklin", "Clinton", "Madison", "Georgetown", "Bristol", "Salem"]

        # Analyze original structure
        has_number = bool(re.search(r"\d", original))
        line_count = len(original.split("\n")) if "\n" in original else 1
        comma_count = original.count(",")

        # Build address matching structure
        number = str(random.randint(1, 999)) if has_number else ""
        street = random.choice(streets)
        city = random.choice(cities)

        if comma_count >= 2 or line_count >= 2:
            # Multi-part address
            if number:
                return f"{number} {street}, {city}"
            return f"{street}, {city}"
        elif comma_count == 1:
            if number:
                return f"{number} {street}, {city}"
            return f"{street}, {city}"
        else:
            # Simple address
            if number:
                return f"{number} {street}"
            return street

    def _generate_tax_id(self, seed: int, original: str) -> str:
        """Generate tax ID preserving format."""
        random.seed(seed)

        # Analyze pattern of original
        pattern = analyze_pattern(original)

        if pattern:
            return generate_from_pattern(pattern, original, seed)
        else:
            # Match original length
            clean = re.sub(r"[^0-9]", "", original)
            length = len(clean) if len(clean) >= 5 else 10
            return "".join([str(random.randint(0, 9)) for _ in range(length)])

    def _generate_national_id(self, seed: int, original: str) -> str:
        """Generate national ID preserving format."""
        random.seed(seed)

        # Analyze pattern of original
        pattern = analyze_pattern(original)

        if pattern:
            # Format-preserving generation
            return generate_from_pattern(pattern, original, seed)
        else:
            # Fallback: match length, first digit non-zero
            clean = re.sub(r"[^0-9]", "", original)
            length = len(clean) if len(clean) >= 5 else 11

            first = str(random.randint(1, 9))
            rest = "".join([str(random.randint(0, 9)) for _ in range(length - 1)])

            return f"{first}{rest}"

    def _generate_passport(self, seed: int, original: str) -> str:
        """Generate passport number preserving format."""
        random.seed(seed)

        # Analyze pattern of original
        pattern = analyze_pattern(original)

        if pattern:
            # Format-preserving generation
            return generate_from_pattern(pattern, original, seed)
        else:
            # Fallback: letter prefix + digits
            prefix = random.choice(string.ascii_uppercase)
            number = "".join([str(random.randint(0, 9)) for _ in range(8)])
            return f"{prefix}{number}"

    def _generate_license_plate(self, seed: int, original: str) -> str:
        """Generate license plate preserving format."""
        random.seed(seed)

        # Analyze pattern of original
        pattern = analyze_pattern(original)

        if pattern:
            # Format-preserving generation
            return generate_from_pattern(pattern, original, seed)
        else:
            # Fallback: generic plate format
            letters = "".join(random.choices(string.ascii_uppercase, k=3))
            numbers = "".join([str(random.randint(0, 9)) for _ in range(4)])
            return f"{letters} {numbers}"

    def _generate_date(self, seed: int, original: str) -> str:
        """Generate date matching original format."""
        fake = self._get_faker(seed)
        date = fake.date_of_birth(minimum_age=18, maximum_age=80)

        # Detect format from original
        if "/" in original:
            return date.strftime("%d/%m/%Y")
        elif "." in original:
            return date.strftime("%d.%m.%Y")
        elif "-" in original:
            return date.strftime("%Y-%m-%d")
        else:
            return date.strftime("%d.%m.%Y")

    def _generate_reference(self, seed: int, original: str) -> str:
        """Generate reference number."""
        random.seed(seed)

        # Match original length
        length = len(original.replace(" ", "").replace("-", ""))
        if length < 6:
            length = 8

        # Mix of letters and numbers
        chars = string.ascii_uppercase + string.digits
        return "".join(random.choices(chars, k=length))

    def _generate_invoice(self, seed: int, original: str) -> str:
        """Generate invoice number preserving format."""
        random.seed(seed)

        # Analyze pattern of original
        pattern = analyze_pattern(original)

        if pattern:
            return generate_from_pattern(pattern, original, seed)
        else:
            # Generic invoice format matching length
            length = len(original) if len(original) >= 6 else 10
            chars = string.ascii_uppercase + string.digits
            return "".join(random.choices(chars, k=length))

    def _generate_signature_placeholder(self, seed: int, original: str) -> str:
        """Generate placeholder for signature."""
        return "[SIGNATURE]"

    def _generate_stamp_placeholder(self, seed: int, original: str) -> str:
        """Generate placeholder for stamp."""
        return "[STAMP]"

    def _generate_generic(self, seed: int, original: str) -> str:
        """Generate generic replacement for unknown types."""
        random.seed(seed)

        # Match original characteristics
        length = len(original)

        if original.isdigit():
            return "".join([str(random.randint(0, 9)) for _ in range(length)])
        elif original.isalpha():
            return "".join(random.choices(string.ascii_uppercase, k=length))
        else:
            # Mix
            chars = string.ascii_uppercase + string.digits
            return "".join(random.choices(chars, k=length))
