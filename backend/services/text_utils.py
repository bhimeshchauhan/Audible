"""
Text processing utilities for cleaning and normalizing text.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

# =========================
# Constants
# =========================

SURROGATE_RE = re.compile(r"[\uD800-\uDFFF]")
POSSESSIVE_RE = re.compile(r"('s|'s)$", re.I)
HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "sir",
    "madam",
    "lady",
    "lord",
    "capt",
    "captain",
    "rev",
    "fr",
    "st",
    "saint",
    "ser",
}

RELATIONSHIP_CATEGORIES: Dict[str, List[str]] = {
    "family": [
        "father",
        "mother",
        "son",
        "daughter",
        "brother",
        "sister",
        "sibling",
        "grandfather",
        "grandmother",
        "grandson",
        "granddaughter",
        "uncle",
        "aunt",
        "nephew",
        "niece",
        "cousin",
        "parent",
        "child",
    ],
    "romantic": [
        "spouse",
        "husband",
        "wife",
        "lover",
        "fiancé",
        "fiancée",
        "partner",
        "romantic interest",
        "admirer",
        "suitor",
        "betrothed",
    ],
    "friendship": ["friend", "best friend", "companion", "confidant", "ally"],
    "professional": [
        "colleague",
        "employer",
        "employee",
        "boss",
        "mentor",
        "student",
        "teacher",
        "master",
        "apprentice",
        "doctor",
        "servant",
    ],
    "social": ["neighbor", "acquaintance", "maid", "butler", "governess"],
    "antagonistic": ["enemy", "rival", "adversary", "nemesis", "opponent"],
}

CATEGORY_COLORS: Dict[str, str] = {
    "family": "#e74c3c",
    "romantic": "#e91e63",
    "friendship": "#3498db",
    "professional": "#9b59b6",
    "social": "#27ae60",
    "antagonistic": "#f39c12",
    "unknown": "#95a5a6",
}


# =========================
# Utility Functions
# =========================


def sanitize_unicode(text: str) -> str:
    """Remove invalid unicode characters from text."""
    if not text:
        return ""
    text = SURROGATE_RE.sub("", text)
    text = "".join(
        ch for ch in text if ch in ("\n", "\t") or (0x20 <= ord(ch) <= 0x10FFFF)
    )
    return text.encode("utf-8", "replace").decode("utf-8", "replace")


def clean_text(s: str) -> str:
    """Clean and normalize text."""
    s = sanitize_unicode(s)
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def log_score(x: int) -> float:
    """Calculate logarithmic score for visualization scaling."""
    return 1.0 + math.log(1 + x)


def get_relationship_category(rel_type: str) -> str:
    """Determine the category of a relationship type."""
    rel_lower = rel_type.lower().strip()
    for category, types in RELATIONSHIP_CATEGORIES.items():
        for t in types:
            if t in rel_lower or rel_lower in t:
                return category
    return "unknown"


def normalize_name(name: str) -> str:
    """Normalize a person's name for consistent identification."""
    name = sanitize_unicode(name).strip()
    name = POSSESSIVE_RE.sub("", name)
    name = re.sub(r"^[\"'" "'']+|[\"'" "'']+$", "", name)
    name = re.sub(r"[^A-Za-z.\- '']+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    parts = name.split()
    if parts and parts[0].rstrip(".").lower() in HONORIFICS:
        parts = parts[1:]
    name = " ".join(parts).strip()
    if len(name) < 2:
        return ""
    return " ".join([p[:1].upper() + p[1:] for p in name.split()])
