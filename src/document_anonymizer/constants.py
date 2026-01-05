"""
Document Anonymization Constants

General-purpose constants and rules for document anonymization.
Defines field types, sensitivity rules, and detection patterns.
"""

from enum import Enum
from typing import Dict, List, Set  # Set used by ALL_LEGAL_SUFFIXES, etc.

# =============================================================================
# ENTITY NAMESPACES
# =============================================================================

class EntityNamespace(Enum):
    """
    Entity namespaces for token generation.

    Provides consistent prefixes for anonymized tokens across documents.
    """
    # Personal entities
    PERSON = "PER"
    ORGANIZATION = "ORG"

    # Contact information
    ADDRESS = "ADR"
    PHONE = "PHN"
    EMAIL = "EML"

    # Identification
    ID_NUMBER = "IDN"
    TAX_ID = "TAX"
    REGISTRATION = "REG"

    # Reference numbers
    REFERENCE = "REF"
    INVOICE = "INV"
    CONTRACT = "CNT"

    # Location
    LOCATION = "LOC"

    # Dates (when sensitive)
    DATE = "DAT"

    # Visual elements
    SIGNATURE = "SIG"
    STAMP = "STP"

    # Unknown/other
    UNKNOWN = "UNK"


# =============================================================================
# DOCUMENT TYPES
# =============================================================================

class DocumentType(Enum):
    """Generic document types."""
    INVOICE = "invoice"
    CONTRACT = "contract"
    LETTER = "letter"
    FORM = "form"
    CERTIFICATE = "certificate"
    REPORT = "report"
    STATEMENT = "statement"
    APPLICATION = "application"
    RECEIPT = "receipt"
    ID_DOCUMENT = "id_document"
    FINANCIAL = "financial"
    LEGAL = "legal"
    MEDICAL = "medical"
    UNKNOWN = "unknown"


# =============================================================================
# LEGAL ENTITY SUFFIXES
# =============================================================================

# Legal entity suffixes by region
LEGAL_ENTITY_SUFFIXES: Dict[str, List[str]] = {
    "english": [
        "LTD", "LTD.", "LIMITED",
        "INC", "INC.", "INCORPORATED",
        "CORP", "CORP.", "CORPORATION",
        "CO", "CO.", "COMPANY",
        "LLC", "L.L.C.",
        "LLP", "L.L.P.",
        "PLC", "P.L.C.",
        "LP", "L.P.",
    ],
    "turkish": [
        "A.S.", "A.Ş.", "AS", "AŞ",
        "LTD", "LTD.", "LTD.ŞTİ.", "LTD. ŞTİ.",
        "ŞTİ", "ŞTİ.",
        "HOLDİNG", "HOLDING",
    ],
    "german": [
        "GMBH", "G.M.B.H.", "GMBH & CO. KG",
        "AG", "A.G.",
        "KG", "K.G.",
        "OHG", "O.H.G.",
    ],
    "french": [
        "SARL", "S.A.R.L.",
        "SA", "S.A.",
        "SAS", "S.A.S.",
        "EURL", "E.U.R.L.",
    ],
    "italian": [
        "SRL", "S.R.L.",
        "SPA", "S.P.A.",
    ],
    "dutch": [
        "BV", "B.V.",
        "NV", "N.V.",
    ],
    "arabic": [
        "WLL", "W.L.L.",
        "LLC", "L.L.C.",
        "FZE", "F.Z.E.",
        "FZC", "F.Z.C.",
        "JSC", "J.S.C.",
    ],
}

# Flattened list of all legal suffixes
ALL_LEGAL_SUFFIXES: Set[str] = set()
for suffixes in LEGAL_ENTITY_SUFFIXES.values():
    ALL_LEGAL_SUFFIXES.update(s.upper() for s in suffixes)


# =============================================================================
# DETECTION PATTERNS
# =============================================================================

# Common patterns for sensitive data detection
# Only patterns with specific formats that won't cause false positives
REFERENCE_PATTERNS: Dict[str, str] = {
    # === Global patterns ===
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ip_address": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    "date_iso": r"\d{4}-\d{2}-\d{2}",
    "date_us": r"\d{1,2}/\d{1,2}/\d{2,4}",
    "date_eu": r"\d{1,2}\.\d{1,2}\.\d{2,4}",

    # === National ID patterns (specific formats only) ===
    # US: Social Security Number (XXX-XX-XXXX) - dash format is unique
    "ssn_us": r"\b\d{3}-\d{2}-\d{4}\b",
    # UK: National Insurance Number (AB 12 34 56 C) - letter+digit+letter format
    "nino_uk": r"\b[A-Z]{2}\s*\d{2}\s*\d{2}\s*\d{2}\s*[A-Z]\b",
    # France: INSEE/NIR (15 digits, starts with 1 or 2) - specific structure
    "insee_fr": r"\b[12]\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{3}\s*\d{3}\s*\d{2}\b",
    # Italy: Codice Fiscale (16 alphanumeric) - very specific pattern
    "cf_it": r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
    # Spain: DNI/NIE (8 digits + letter or X/Y/Z + 7 digits + letter)
    "dni_es": r"\b(\d{8}[A-Z]|[XYZ]\d{7}[A-Z])\b",

    # === License plate patterns (specific formats only) ===
    # Turkey: 34 ABC 1234 (city code + letters + numbers)
    "plate_tr": r"\b\d{2}\s*[A-Z]{1,3}\s*\d{2,4}\b",
    # UK: AB12 CDE (very specific format)
    "plate_uk": r"\b[A-Z]{2}\d{2}\s*[A-Z]{3}\b",
    # France: AA-123-AA (dash-separated format)
    "plate_fr": r"\b[A-Z]{2}-\d{3}-[A-Z]{2}\b",
    # Italy: AB 123 CD (letter-number-letter pattern)
    "plate_it": r"\b[A-Z]{2}\s*\d{3}\s*[A-Z]{2}\b",
    # Spain: 1234 ABC (4 digits + 3 letters)
    "plate_es": r"\b\d{4}\s*[A-Z]{3}\b",
    # Russia: A123BC 77 (letter + digits + letters + region)
    "plate_ru": r"\b[A-Z]\d{3}[A-Z]{2}\s*\d{2,3}\b",

    # === Postal code patterns (only UK - others are just digits) ===
    # UK: SW1A 1AA (letter+digit+letter format is unique)
    "postal_uk": r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b",

    # === Phone patterns (country code prefix required) ===
    # US/Canada: +1 XXX XXX XXXX
    "phone_us": r"\+1\s*\(?\d{3}\)?\s*\d{3}\s*\d{4}",
    # Turkey: +90 XXX XXX XX XX
    "phone_tr": r"\+90\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}",
    # Germany: +49 XXX XXXXXXXX
    "phone_de": r"\+49\s*\d{3,4}\s*\d{6,8}",
    # France: +33 X XX XX XX XX
    "phone_fr": r"\+33\s*\d\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{2}",
    # UK: +44 XXXX XXXXXX
    "phone_uk": r"\+44\s*\d{4}\s*\d{6}",
    # Russia: +7 XXX XXX XX XX
    "phone_ru": r"\+7\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}",
}


# =============================================================================
# VERIFICATION PATTERNS
# =============================================================================

# Patterns to check for in masked documents (potential leaks)
VERIFICATION_SENSITIVE_PATTERNS: Dict[str, List[str]] = {
    "email": [r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"],
    "ssn": [r"\d{3}-\d{2}-\d{4}"],
    "phone": [r"\+?[1-9]\d{6,14}"],
}

# Expected patterns for masked/dummy text
EXPECTED_MASKED_PATTERNS: List[str] = [
    r"[A-Z]{3}-[A-F0-9]{8}",  # Token format: XXX-XXXXXXXX
    r"\[MASKED\]",
    r"\[REDACTED\]",
    r"X{4,}",
    r"\*{4,}",
]

# Generic document phrases (not sensitive)
GENERIC_DOCUMENT_PHRASES: Set[str] = {
    "date", "page", "total", "amount", "quantity",
    "description", "price", "tax", "subtotal",
    "terms and conditions", "please", "thank you",
    "sincerely", "regards", "dear", "reference",
}

# Boilerplate patterns (not sensitive)
BOILERPLATE_PATTERNS: List[str] = [
    r"all rights reserved",
    r"confidential",
    r"for official use only",
    r"please review",
    r"subject to",
    r"terms and conditions",
]


# =============================================================================
# FIELD TYPE MAPPING
# =============================================================================

# Field type keyword to namespace mapping
_FIELD_TYPE_MAPPING: Dict[str, EntityNamespace] = {
    "person": EntityNamespace.PERSON,
    "name": EntityNamespace.PERSON,
    "first": EntityNamespace.PERSON,
    "last": EntityNamespace.PERSON,
    "company": EntityNamespace.ORGANIZATION,
    "organization": EntityNamespace.ORGANIZATION,
    "business": EntityNamespace.ORGANIZATION,
    "corp": EntityNamespace.ORGANIZATION,
    "address": EntityNamespace.ADDRESS,
    "street": EntityNamespace.ADDRESS,
    "postal": EntityNamespace.ADDRESS,
    "phone": EntityNamespace.PHONE,
    "tel": EntityNamespace.PHONE,
    "fax": EntityNamespace.PHONE,
    "mobile": EntityNamespace.PHONE,
    "email": EntityNamespace.EMAIL,
    "id": EntityNamespace.ID_NUMBER,
    "ssn": EntityNamespace.ID_NUMBER,
    "passport": EntityNamespace.ID_NUMBER,
    "license": EntityNamespace.ID_NUMBER,
    "tax": EntityNamespace.TAX_ID,
    "vat": EntityNamespace.TAX_ID,
    "registration": EntityNamespace.REGISTRATION,
    "reference": EntityNamespace.REFERENCE,
    "ref": EntityNamespace.REFERENCE,
    "number": EntityNamespace.REFERENCE,
    "invoice": EntityNamespace.INVOICE,
    "contract": EntityNamespace.CONTRACT,
    "signature": EntityNamespace.SIGNATURE,
    "sign": EntityNamespace.SIGNATURE,
    "stamp": EntityNamespace.STAMP,
    "seal": EntityNamespace.STAMP,
    "date": EntityNamespace.DATE,
}


def get_namespace_for_field(field_type: str) -> EntityNamespace:
    """Map field type to entity namespace."""
    field_type_lower = field_type.lower()
    for keyword, namespace in _FIELD_TYPE_MAPPING.items():
        if keyword in field_type_lower:
            return namespace
    return EntityNamespace.UNKNOWN


def has_legal_suffix(text: str) -> bool:
    """
    Check if text contains a legal entity suffix.

    Only matches:
    - Text ending with suffix (e.g., "Acme Ltd")
    - Text with suffix followed by punctuation (e.g., "Acme Ltd.")
    - Text with suffix at word boundary

    Avoids false positives like "Air conditioning" matching "IN".

    Args:
        text: Text to check

    Returns:
        True if legal suffix found
    """
    import re

    text_upper = text.upper().strip()

    # Must be relatively short (typical company names are < 100 chars)
    if len(text_upper) > 100:
        return False

    # Skip if text looks like a sentence (has multiple spaces and common words)
    lower_text = text.lower()
    sentence_indicators = ['you', 'the', 'for', 'and', 'can', 'will', 'with', 'this', 'that']
    if sum(1 for ind in sentence_indicators if f' {ind} ' in lower_text) >= 2:
        return False

    # Suffixes that are too short and cause false positives
    short_suffixes = {'CO', 'CO.', 'IN', 'AG', 'A.G.', 'KG', 'K.G.', 'LP', 'L.P.', 'BV', 'B.V.', 'NV', 'N.V.', 'SA', 'S.A.'}

    for suffix in ALL_LEGAL_SUFFIXES:
        # Skip very short suffixes - too many false positives
        if suffix in short_suffixes:
            continue

        # Must end with suffix or have suffix at word boundary
        if text_upper.endswith(suffix):
            # Ensure there's a word before the suffix (not just the suffix alone)
            prefix = text_upper[:-len(suffix)].strip()
            if len(prefix) >= 2:
                return True

        # Check for suffix followed by punctuation or end of text
        # e.g., "Acme Ltd." or "Acme, Ltd"
        pattern = rf'\b[A-Z][A-Z0-9&\s]+\s+{re.escape(suffix)}(?:[.,]|\s|$)'
        if re.search(pattern, text_upper):
            return True

    return False
