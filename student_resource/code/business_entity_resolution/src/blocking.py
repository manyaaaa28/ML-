"""
Candidate generation (blocking) - recall-first, union of cheap strategies.

Given normalized S1 and S2/S3 frames (see preprocess.normalize_frame), build
a per-S1-entity candidate set as the union of:

  [A] token-inverted-index lookup on core name tokens
  [B] char n-gram TF-IDF cosine top-k (sparse_dot_topn)
  [C] sorted-neighborhood exact-match on locality tokens + name-prefix
  [D] address-token-index lookup (independent of name)

All four operate on integer row positions internally (not string IDs) to
keep memory down, and are combined with a plain set union per S1 row.

Each strategy caps ITS OWN contribution per S1 entity (B via `top_n` in
sp_matmul_topn, C via `window`, A/D via `per_entity_cap` ranked by shared-
token count) rather than leaving any one strategy free to blow up the
union - see token_index_candidates' docstring for why this is done at the
source instead of as a separate post-hoc pruning pass (an earlier, removed
`prune_candidates_topk` re-ranked the whole union by TF-IDF cosine after
the fact: too slow at scale - CSR sub-matrix extraction per S1 row - and
biased against exactly the candidates strategies C/D exist to catch, i.e.
locality/address matches with low name similarity).
"""

from __future__ import annotations

import gc
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # pragma: no cover
    sp_matmul_topn = None


# --------------------------------------------------------------------------
# [A] Token-inverted-index blocking
# --------------------------------------------------------------------------

def build_token_index(core_token_lists: List[list]) -> Dict[str, List[int]]:
    """token -> list of row positions whose name_core_tokens contain it."""
    index: Dict[str, List[int]] = defaultdict(list)
    for row_pos, tokens in enumerate(core_token_lists):
        for tok in set(tokens):
            if len(tok) < 2:  # skip single-char tokens, too common/noisy
                continue
            index[tok].append(row_pos)
    return index


def token_index_candidates(
    s1_core_token_lists: List[list],
    index: Dict[str, List[int]],
    max_postings_per_token: int = 5000,
    per_entity_cap: Optional[int] = 300,
) -> List[Set[int]]:
    """For each S1 row, ref rows sharing its core tokens, capped per entity.

    ``max_postings_per_token`` guards against a single very-common token
    (e.g. a generic word) blowing up the candidate set for every S1 row that
    contains it - tokens with more postings than this are skipped as
    uninformative for blocking (still covered by TF-IDF/locality strategies).

    ``per_entity_cap`` bounds this strategy's OWN contribution per S1 row,
    ranked by how many distinct core tokens a ref row shares (more shared
    tokens = a stronger match under this strategy's own signal, not a
    foreign one) - this is what keeps a plain union of postings from
    blowing up to thousands of candidates for entities whose tokens are
    individually common but collectively distinctive (see match.py's or
    pipeline.py's module docstring on why capping at the SOURCE, per
    strategy, beats a post-hoc re-ranking pass). ``None`` disables the cap
    (full union, the old behavior).
    """
    out: List[Set[int]] = []
    for tokens in s1_core_token_lists:
        counts: Counter = Counter()
        for tok in set(tokens):
            postings = index.get(tok)
            if not postings or len(postings) > max_postings_per_token:
                continue
            counts.update(postings)
        if per_entity_cap is None or len(counts) <= per_entity_cap:
            cands = set(counts.keys())
        else:
            cands = {p for p, _ in counts.most_common(per_entity_cap)}
        out.append(cands)
    return out


# --------------------------------------------------------------------------
# [B] Char n-gram TF-IDF top-k cosine
# --------------------------------------------------------------------------

def fit_char_ngram_vectorizer(
    corpus: List[str], ngram_range=(3, 5), max_features: int = 200_000
) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
    )
    vec.fit(corpus)
    return vec


def tfidf_topk_candidates(
    s1_texts: List[str],
    ref_texts: List[str],
    vectorizer: TfidfVectorizer,
    top_n: int = 15,
    lower_bound: float = 0.35,
    s1_chunk_size: int = 200_000,
) -> List[Set[int]]:
    """Char-ngram TF-IDF cosine top-k, chunked over S1 rows for memory safety.

    Returns, per S1 row, the set of reference row positions among its top-k
    cosine matches with score >= ``lower_bound``.
    """
    if sp_matmul_topn is None:
        raise ImportError("sparse_dot_topn is required for tfidf_topk_candidates")

    ref_mat = vectorizer.transform(ref_texts).astype(np.float32)
    ref_mat_t = ref_mat.T.tocsr()

    out: List[Set[int]] = [set() for _ in range(len(s1_texts))]
    n = len(s1_texts)
    for start in range(0, n, s1_chunk_size):
        end = min(start + s1_chunk_size, n)
        chunk_mat = vectorizer.transform(s1_texts[start:end]).astype(np.float32)
        sim = sp_matmul_topn(
            chunk_mat, ref_mat_t, top_n=top_n, threshold=lower_bound, sort=False
        )
        sim = sim.tocsr()
        for local_i in range(sim.shape[0]):
            row = sim.getrow(local_i)
            if row.nnz:
                out[start + local_i] = set(row.indices.tolist())
        del chunk_mat, sim
        gc.collect()
    return out


# --------------------------------------------------------------------------
# [C] Sorted-neighborhood on locality tokens + name prefix
# --------------------------------------------------------------------------

def build_sorted_neighborhood_index(
    locality_strings: List[str], name_norm_strings: List[str]
) -> Dict[str, List[int]]:
    """Key = (one locality token) + first-3-chars name prefix -> row positions.

    ``locality_strings`` holds a *bag* of tokens (see preprocess.extract_locality
    - city/state/region, order-independent), so we index every token in the
    bag, not just one arbitrarily chosen one; a row is reachable through any
    of its locality tokens paired with its name prefix. A cheap,
    order-tolerant near-exact block: two records in roughly the same place
    whose normalized name starts the same way are almost certainly worth
    comparing, even if TF-IDF/token blocking missed them due to typos
    elsewhere in the name.
    """
    index: Dict[str, List[int]] = defaultdict(list)
    for row_pos, (loc, name) in enumerate(zip(locality_strings, name_norm_strings)):
        name_prefix = name[:3] if name else ""
        loc_tokens = loc.split(" ") if loc else []
        if not loc_tokens:
            loc_tokens = [""]
        for loc_tok in loc_tokens:
            key = f"{loc_tok}|{name_prefix}"
            index[key].append(row_pos)
    return index


def sorted_neighborhood_candidates(
    s1_locality_strings: List[str],
    s1_name_norm_strings: List[str],
    index: Dict[str, List[int]],
    window: int = 10,
) -> List[Set[int]]:
    out: List[Set[int]] = []
    for loc, name in zip(s1_locality_strings, s1_name_norm_strings):
        name_prefix = name[:3] if name else ""
        loc_tokens = loc.split(" ") if loc else [""]
        cands: Set[int] = set()
        for loc_tok in loc_tokens:
            key = f"{loc_tok}|{name_prefix}"
            postings = index.get(key, [])
            cands.update(postings[:window] if len(postings) > window else postings)
        out.append(cands)
    return out


# --------------------------------------------------------------------------
# [D] Address-token blocking (independent of name - catches heavily
#     name-corrupted records that still share address tokens)
# --------------------------------------------------------------------------

def build_address_token_index(addr_norm_strings: List[str]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for row_pos, addr in enumerate(addr_norm_strings):
        if not addr:
            continue
        for tok in set(addr.split(" ")):
            if len(tok) < 4 or tok.isdigit():
                # skip short/common words and pure numbers (house numbers
                # are noisy/unreliable here - see preprocess.py's EDA note)
                continue
            index[tok].append(row_pos)
    return index


def address_token_candidates(
    s1_addr_norm_strings: List[str],
    index: Dict[str, List[int]],
    max_postings_per_token: int = 5000,
    per_entity_cap: Optional[int] = 300,
) -> List[Set[int]]:
    """Same shared-token-count ranking + per-entity cap as
    ``token_index_candidates`` above - see its docstring. This strategy is
    usually the single largest contributor to candidate-set bloat (locality
    words like a city name are shared by very many ref rows), so the cap
    matters most here.
    """
    out: List[Set[int]] = []
    for addr in s1_addr_norm_strings:
        counts: Counter = Counter()
        if addr:
            for tok in set(addr.split(" ")):
                postings = index.get(tok)
                if not postings or len(postings) > max_postings_per_token:
                    continue
                counts.update(postings)
        if per_entity_cap is None or len(counts) <= per_entity_cap:
            cands = set(counts.keys())
        else:
            cands = {p for p, _ in counts.most_common(per_entity_cap)}
        out.append(cands)
    return out


# --------------------------------------------------------------------------
# Union orchestration
# --------------------------------------------------------------------------

def union_candidate_sets(*candidate_lists: List[Set[int]]) -> List[Set[int]]:
    n = len(candidate_lists[0])
    out: List[Set[int]] = [set() for _ in range(n)]
    for cand_list in candidate_lists:
        for i, s in enumerate(cand_list):
            out[i] |= s
    return out


def positions_to_ids(candidate_positions: List[Set[int]], id_array: np.ndarray) -> List[Set[str]]:
    return [set(id_array[list(pos)]) if pos else set() for pos in candidate_positions]
