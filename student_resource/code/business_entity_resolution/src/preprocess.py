"""
Script-agnostic normalization for business names, addresses and country.

Nothing here is hardcoded to a specific country. The legal-suffix dictionary
is mined from the data itself (frequency + trailing-position signal), so it
picks up US/India suffixes from training data and French suffixes (SARL,
SAS, SA, EURL, ...) from the test set the same way, without a country
allowlist anywhere in the normalization path.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional

import pandas as pd

try:
    from anyascii import anyascii
except ImportError:  # pragma: no cover - anyascii is a hard requirement
    def anyascii(text: str) -> str:  # type: ignore
        return text

# --------------------------------------------------------------------------
# Name normalization
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_AMP_RE = re.compile(r"&")

# Generic English/URL-junk stopword-ish tokens seen in the noise patterns
# (site fragments, pipes-as-separators, leading dashes). These are stripped
# as junk tokens, not as a language-specific stopword list.
_JUNK_TOKENS = {"www", "com", "http", "https", "inc.com"}


def fold_unicode(text: str) -> str:
    """NFKD-normalize then ASCII-fold via anyascii (script-agnostic)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return anyascii(nfkd)


def clean_name(raw: str) -> str:
    """Lowercase, ASCII-fold, expand '&', strip punctuation, collapse whitespace."""
    if not raw:
        return ""
    text = fold_unicode(raw)
    text = text.lower()
    text = _AMP_RE.sub(" and ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list:
    if not text:
        return []
    return [t for t in text.split(" ") if t and t not in _JUNK_TOKENS]


# --------------------------------------------------------------------------
# Data-driven legal-suffix mining
# --------------------------------------------------------------------------

def mine_legal_suffixes(
    name_token_lists: Iterable[list],
    min_count: int = 200,
    min_trailing_ratio: float = 0.5,
    max_suffix_tokens: int = 2,
) -> set:
    """Mine trailing-position tokens/bigrams that behave like legal suffixes.

    For each candidate token (or bigram), we compare how often it appears as
    the LAST token(s) of a name vs. how often it appears anywhere. A token
    that is disproportionately trailing (e.g. "ltd", "llc", "sarl", "pvt")
    is treated as a legal-suffix token. This generalizes to unseen countries
    (France's SARL/SAS/EURL) because it only looks at positional statistics
    in whatever corpus it's given — call it on train+test combined.
    """
    total_count: Counter = Counter()
    trailing1_count: Counter = Counter()
    trailing2_count: Counter = Counter()

    for tokens in name_token_lists:
        if not tokens:
            continue
        total_count.update(tokens)
        trailing1_count[tokens[-1]] += 1
        if len(tokens) >= 2:
            trailing2_count[(tokens[-2], tokens[-1])] += 1

    suffixes: set = set()

    for tok, trail_n in trailing1_count.items():
        total_n = total_count[tok]
        if total_n >= min_count and (trail_n / total_n) >= min_trailing_ratio:
            suffixes.add(tok)

    if max_suffix_tokens >= 2:
        for (t1, t2), trail_n in trailing2_count.items():
            # Only keep a bigram suffix if it's meaningfully more common than
            # chance overlap with the unigram suffixes already found, and
            # occurs often enough on its own (e.g. "pvt ltd", "private limited").
            bigram_total = min(total_count[t1], total_count[t2])
            if bigram_total >= min_count and (trail_n / max(bigram_total, 1)) >= min_trailing_ratio:
                suffixes.add(f"{t1} {t2}")

    return suffixes


def strip_suffix_tokens(tokens: list, suffix_vocab: set, max_strip: int = 3) -> list:
    """Repeatedly drop trailing tokens/bigrams found in ``suffix_vocab``."""
    if not tokens:
        return tokens
    tokens = list(tokens)
    stripped = 0
    while tokens and stripped < max_strip:
        if len(tokens) >= 2 and f"{tokens[-2]} {tokens[-1]}" in suffix_vocab:
            tokens = tokens[:-2]
            stripped += 2
            continue
        if tokens[-1] in suffix_vocab:
            tokens = tokens[:-1]
            stripped += 1
            continue
        break
    return tokens


# --------------------------------------------------------------------------
# Address normalization
# --------------------------------------------------------------------------

_ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "apt": "apartment", "flt": "flat", "bldg": "building", "twp": "township",
    "ct": "court", "pl": "place", "sq": "square", "ter": "terrace",
    "pkwy": "parkway", "hts": "heights", "mnt": "mount", "fwy": "freeway",
}

# Pure-digit postal codes cover US ZIP (5), India PIN (6) and France (5)
# without hardcoding any single country's format; digit runs 4-8 long catch
# essentially every real-world postal code we could plausibly see.
#
# EDA finding (see plan doc): this dataset's business_address field almost
# never actually contains a real postal/PIN/ZIP code (~35-37% of rows have
# ANY 4-8 digit run at all, and manual inspection shows those runs are
# overwhelmingly house/plot/unit numbers, e.g. "KH NO. -570/13", "8219 17RD
# STREET" - not postal codes). So `postal` below is kept as a *weak,
# secondary* feature, not a primary blocking key: it usually just re-derives
# something close to `street_number`. Blocking instead leans on name tokens,
# TF-IDF, and locality (city/state/region), which are far more reliably
# present in this data.
_POSTAL_RE = re.compile(r"\b\d{4,8}\b")
_LEADING_NUM_RE = re.compile(r"\b(\d{1,6}[a-zA-Z]?)\b")

_STREET_KEYWORDS = {
    "road", "street", "avenue", "lane", "drive", "highway", "boulevard",
    "court", "place", "square", "terrace", "parkway", "way", "circle",
    "trail", "walk", "path", "row", "close", "crescent", "freeway",
}
_UNIT_KEYWORDS = {
    "unit", "apt", "apartment", "suite", "ste", "flat", "floor", "fl",
    "building", "bldg", "block", "plot", "sector", "near", "po", "box",
}


def clean_address(raw: str) -> str:
    if not raw:
        return ""
    text = fold_unicode(raw)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = [_ADDR_ABBREV.get(t, t) for t in text.split()]
    text = " ".join(tokens)
    text = _WS_RE.sub(" ", text).strip()
    return text


def extract_postal(raw_or_clean: str) -> str:
    """Best-effort generic postal-code token: first 4-8 digit run found."""
    if not raw_or_clean:
        return ""
    # Collapse "400 001" -> "400001" style split codes before matching.
    compact = re.sub(r"(?<=\d) (?=\d)", "", raw_or_clean)
    m = _POSTAL_RE.search(compact)
    return m.group(0) if m else ""


def _segment_is_street_like(raw_segment: str) -> bool:
    if any(ch.isdigit() for ch in raw_segment):
        return True
    lowered = raw_segment.lower()
    return any(kw in lowered.split() for kw in _STREET_KEYWORDS)


def _segment_is_unit_like(raw_segment: str) -> bool:
    lowered = raw_segment.lower()
    return any(kw in lowered.split() for kw in _UNIT_KEYWORDS)


def extract_locality(raw_address: str, country: str) -> str:
    """Bag-of-tokens locality signal, tolerant of component reordering.

    Component order varies row to row in this dataset (state-first,
    street-last, city-and-region-swapped, etc. - an explicit noise pattern
    per the problem statement), so picking a single "the locality segment"
    by position is fragile either way round. Instead we drop the country
    segment plus any segment that looks like a street (has a digit or a
    street-suffix word) or a unit/building descriptor, then return the
    UNION of tokens from every remaining segment (city, state/region,
    whatever's left), sorted for stable comparison. Two addresses that list
    "Mumbai, Maharashtra" vs "Maharashtra, Mumbai" - or one with only the
    city and the other with only the region - still overlap on shared
    tokens under a Jaccard/token-overlap comparison, which is what this
    feeds into downstream (features.py), rather than needing an exact
    single-segment string match.
    """
    if not raw_address:
        return ""
    segments = [s.strip() for s in raw_address.split(",") if s.strip()]
    if not segments:
        return ""
    country_norm = (country or "").strip().lower()

    non_country = [s for s in segments if s.lower() != country_norm]
    pool = non_country if non_country else segments

    candidates = [
        s for s in pool
        if not _segment_is_street_like(s) and not _segment_is_unit_like(s)
    ]
    if not candidates:
        candidates = pool

    tokens: set = set()
    for seg in candidates:
        tokens.update(tokenize(clean_name(seg)))
    return " ".join(sorted(tokens))


def extract_street_number(raw_address: str) -> str:
    if not raw_address:
        return ""
    m = _LEADING_NUM_RE.search(raw_address)
    return m.group(1).lower() if m else ""


# --------------------------------------------------------------------------
# Country normalization (open set, generic fallback by construction)
# --------------------------------------------------------------------------

def normalize_country(raw: str) -> str:
    if not raw:
        return ""
    return clean_name(raw)


# --------------------------------------------------------------------------
# Frame-level normalization entry point
# --------------------------------------------------------------------------

def normalize_frame(df: pd.DataFrame, suffix_vocab: Optional[set] = None) -> pd.DataFrame:
    """Add normalized columns to a source frame (entity_id/business_name/business_address/country).

    Returns a new frame with:
      name_norm, name_tokens (list), name_core (sorted core tokens after
      suffix-strip, joined by space for storage), addr_norm, postal,
      locality, street_number, country_norm
    """
    out = df.copy()
    out["name_norm"] = out["business_name"].map(clean_name)
    out["name_tokens"] = out["name_norm"].map(tokenize)

    if suffix_vocab:
        out["name_core_tokens"] = out["name_tokens"].map(
            lambda toks: strip_suffix_tokens(toks, suffix_vocab)
        )
    else:
        out["name_core_tokens"] = out["name_tokens"]

    out["name_core"] = out["name_core_tokens"].map(lambda toks: " ".join(sorted(set(toks))))

    out["addr_norm"] = out["business_address"].map(clean_address)
    out["postal"] = out["addr_norm"].map(extract_postal)
    out["country_norm"] = out["country"].map(normalize_country)
    out["locality"] = [
        extract_locality(addr, ctry)
        for addr, ctry in zip(out["business_address"], out["country"])
    ]
    out["street_number"] = out["business_address"].map(extract_street_number)

    return out
