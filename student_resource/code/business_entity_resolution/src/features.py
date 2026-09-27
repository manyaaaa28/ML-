"""
Pairwise feature computation for (S1, candidate) pairs.

Operates on a batch of pairs at a time (row-position indices into the S1 and
reference frames), returns a float32 feature matrix. Never materializes the
full feature matrix for every candidate pair at once - the caller
(pipeline.py) iterates in batches of a few hundred thousand pairs.

Performance note: an earlier version of this module rebuilt token sets and
ran a pure-Python O(len(a)*len(b)) longest-common-substring DP inside the
per-pair loop - benchmarked at ~2,900 pairs/sec, which projects to ~1 hour
per 10M pairs and would make both the dev loop and any full-scale run
impractical. Two changes fixed that:
  1. ``precompute_frame_cache`` builds every row's token sets/splits ONCE
     per frame (called once per batch of pairs, not once per pair) - a
     given S1 row is matched against many candidates, so this turns
     O(n_pairs) set-construction into O(frame_size).
  2. The substring DP is replaced with rapidfuzz's C-implemented longest
     common SUBSEQUENCE (LCSseq) - not identical semantics, but the same
     "how much do these strings share in order" signal, at C speed instead
     of a Python nested loop.
"""

from __future__ import annotations

from typing import List

import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler, LCSseq

FEATURE_NAMES = [
    # name features
    "name_levenshtein_norm",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_dice",
    "name_containment_jaccard",
    "name_tfidf_cosine",
    "name_lcs_ratio",
    "name_suffix_exact_match",
    # address features
    "addr_token_jaccard",
    "postal_exact_match",
    "locality_token_jaccard",
    "addr_levenshtein_norm",
    "street_number_match",
    # meta features
    "country_match",
    "source_is_s2",
    "name_len_ratio",
    "shared_token_count",
]

_DTYPE = np.float32


def _safe_ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def _set_jaccard(sa: frozenset, sb: frozenset) -> float:
    # Both-empty deliberately returns 0.0, not 1.0: mutual absence of data
    # (e.g. two names/addresses that both reduce to no tokens) is
    # uninformative, not evidence of a match - scoring it as a perfect
    # match would hand the classifier a false-positive-generating shortcut,
    # which is exactly wrong for a precision-heavy metric.
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return _safe_ratio(inter, len(sa | sb))


def _set_dice(sa: frozenset, sb: frozenset) -> float:
    if not sa or not sb:
        return 0.0
    return _safe_ratio(2 * len(sa & sb), len(sa) + len(sb))


def _set_containment(sa: frozenset, sb: frozenset) -> float:
    """Asymmetric overlap: |A∩B| / min(|A|,|B|) - handles DBA / added trade-name tokens.

    Standard Jaccard penalizes "Acme Corp" vs "Acme Corp DBA Best Pizza"
    heavily even though one name fully contains the other; containment
    measures how much of the SHORTER name's tokens are covered.
    """
    if not sa or not sb:
        return 0.0
    return _safe_ratio(len(sa & sb), min(len(sa), len(sb)))


def precompute_frame_cache(frame) -> dict:
    """Precompute per-row derived structures ONCE per frame.

    Call this once per (frame, batch-of-chunks) rather than inside the
    per-pair loop - see module docstring for why this matters.

    Uses ``.tolist()`` rather than ``.values`` to pull columns out of the
    frame: pandas 3.x defaults string columns to a PyArrow-backed
    ArrowExtensionArray, whose per-element ``__getitem__`` does real
    type-checking work (profiled at ~14s of a 30s run for 1.1M accesses).
    A plain Python list has O(1) pointer-following access with none of
    that overhead, which matters a lot here since the per-pair loop below
    indexes into these arrays millions of times.
    """
    name = frame["name_norm"].tolist()
    core = frame["name_core_tokens"].tolist()
    core_sets = [frozenset(t) for t in core]

    addr = frame["addr_norm"].tolist()
    addr_sets = [frozenset(a.split(" ")) if a else frozenset() for a in addr]

    postal = frame["postal"].tolist()

    locality = frame["locality"].tolist()
    locality_sets = [frozenset(l.split(" ")) if l else frozenset() for l in locality]

    street = frame["street_number"].tolist()
    country = frame["country_norm"].tolist()

    entity_id = frame["entity_id"].tolist()
    is_s2 = [str(e).startswith("S2-") for e in entity_id]

    name_len = [len(n) for n in name]

    return dict(
        name=name, core_sets=core_sets, addr=addr, addr_sets=addr_sets,
        postal=postal, locality_sets=locality_sets, street=street,
        country=country, is_s2=is_s2, name_len=name_len,
    )


def compute_pair_features_batch(
    s1_idx: np.ndarray,
    ref_idx: np.ndarray,
    s1_cache: dict,
    ref_cache: dict,
    tfidf_cosine: np.ndarray = None,
) -> np.ndarray:
    """Compute the feature matrix for a batch of (s1_row, ref_row) pairs.

    ``s1_cache``/``ref_cache`` come from ``precompute_frame_cache`` - build
    them once outside any per-chunk loop and reuse across every batch drawn
    from the same frame.
    ``tfidf_cosine`` is an optional precomputed cosine-similarity array
    aligned with the batch (reused from blocking's TF-IDF pass instead of
    recomputing), else 0.0 is used for that feature.
    """
    n = len(s1_idx)
    feats = np.zeros((n, len(FEATURE_NAMES)), dtype=_DTYPE)

    s1_name, ref_name = s1_cache["name"], ref_cache["name"]
    s1_core_sets, ref_core_sets = s1_cache["core_sets"], ref_cache["core_sets"]
    s1_addr_sets, ref_addr_sets = s1_cache["addr_sets"], ref_cache["addr_sets"]
    s1_postal, ref_postal = s1_cache["postal"], ref_cache["postal"]
    s1_loc_sets, ref_loc_sets = s1_cache["locality_sets"], ref_cache["locality_sets"]
    s1_street, ref_street = s1_cache["street"], ref_cache["street"]
    s1_country, ref_country = s1_cache["country"], ref_cache["country"]
    s1_len, ref_len = s1_cache["name_len"], ref_cache["name_len"]
    ref_is_s2 = ref_cache["is_s2"]

    for k in range(n):
        i, j = int(s1_idx[k]), int(ref_idx[k])
        nA, nB = s1_name[i], ref_name[j]
        sA, sB = s1_core_sets[i], ref_core_sets[j]

        if nA and nB:
            feats[k, 0] = 1.0 - Levenshtein.distance(nA, nB) / max(s1_len[i], ref_len[j])
            feats[k, 1] = JaroWinkler.similarity(nA, nB)
            feats[k, 6] = LCSseq.normalized_similarity(nA, nB)
        feats[k, 2] = _set_jaccard(sA, sB)
        feats[k, 3] = _set_dice(sA, sB)
        feats[k, 4] = _set_containment(sA, sB)
        feats[k, 5] = float(tfidf_cosine[k]) if tfidf_cosine is not None else 0.0
        feats[k, 7] = 1.0 if (sA and sB and sA == sB) else 0.0

        aA, aB = s1_addr_sets[i], ref_addr_sets[j]
        feats[k, 8] = _set_jaccard(aA, aB)
        pA, pB = s1_postal[i], ref_postal[j]
        feats[k, 9] = 1.0 if (pA and pB and pA == pB) else 0.0
        feats[k, 10] = _set_jaccard(s1_loc_sets[i], ref_loc_sets[j])
        addrA_str, addrB_str = s1_cache["addr"][i], ref_cache["addr"][j]
        if addrA_str and addrB_str:
            feats[k, 11] = 1.0 - Levenshtein.distance(addrA_str, addrB_str) / max(len(addrA_str), len(addrB_str))
        strA, strB = s1_street[i], ref_street[j]
        feats[k, 12] = 1.0 if (strA and strB and strA == strB) else 0.0

        feats[k, 13] = 1.0 if (s1_country[i] and s1_country[i] == ref_country[j]) else 0.0
        feats[k, 14] = 1.0 if ref_is_s2[j] else 0.0
        if nA and nB:
            feats[k, 15] = min(s1_len[i], ref_len[j]) / max(s1_len[i], ref_len[j])
        feats[k, 16] = float(len(sA & sB))

    return feats
