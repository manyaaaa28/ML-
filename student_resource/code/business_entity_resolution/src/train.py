"""
Build training pairs (positives + capped hard negatives), train a LightGBM
pairwise match classifier, and calibrate its output probabilities.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

DEFAULT_LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    max_depth=7,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=500,
    min_child_samples=20,
    n_jobs=-1,
    verbosity=-1,
)


def _set_jaccard(sa: set, sb: set) -> float:
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_training_pairs(
    s1_frame,
    ref_frame,
    candidate_positions: List[Set[int]],
    gt_map: Dict[str, Set[str]],
    top_k_neg: int = 4,
    neg_pos_ratio: float = 6.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (s1_idx, ref_idx, labels) arrays for the training set.

    Positives: every blocked candidate that IS a ground-truth match.

    Negatives: per S1 entity, the top ``top_k_neg`` non-matching blocked
    candidates from EACH of two hardness rankings - name-core-token Jaccard
    AND locality-token Jaccard - unioned (deduped), not just one.

    An earlier version ranked "hardest negative" by name-token similarity
    alone. That systematically excluded exactly the failure mode blocking
    is prone to (see match.py's module docstring on why blocking casts a
    wide net): two DIFFERENT businesses that share a city/state/country and
    a common legal-suffix pattern but have unrelated names - such a
    candidate ranks as "easy" under name-only similarity (near-zero token
    overlap) and so never got selected as a training negative, meaning the
    model never saw this exact confusion during training and over-trusted
    locality/country agreement at inference. Ranking by locality similarity
    too, and taking candidates hard under EITHER criterion, fixes this.
    This covers singleton entities as well (all their candidates become
    negatives), which is exactly what protects singleton precision. A
    global ``neg_pos_ratio`` cap then subsamples down if the combined
    hard-negative pool still dwarfs the positive count.
    """
    # .tolist(), not .values: pandas 3.x's PyArrow-backed columns have a slow
    # per-element __getitem__, and this function indexes into these arrays
    # once per candidate across every S1 row - see features.py's
    # precompute_frame_cache docstring for the profiling behind this.
    s1_ids = s1_frame["entity_id"].tolist()
    ref_ids = ref_frame["entity_id"].tolist()
    s1_core = [set(t) for t in s1_frame["name_core_tokens"].tolist()]
    ref_core = [set(t) for t in ref_frame["name_core_tokens"].tolist()]
    s1_loc = [set(l.split(" ")) if l else set() for l in s1_frame["locality"].tolist()]
    ref_loc = [set(l.split(" ")) if l else set() for l in ref_frame["locality"].tolist()]

    pos_pairs: List[Tuple[int, int]] = []
    neg_pairs: List[Tuple[int, int]] = []

    for i in range(len(s1_ids)):
        cand_positions = candidate_positions[i]
        if not cand_positions:
            continue
        true_ids = gt_map.get(s1_ids[i], set())
        pos_here, neg_here = [], []
        for p in cand_positions:
            if ref_ids[p] in true_ids:
                pos_here.append(p)
            else:
                neg_here.append(p)
        pos_pairs.extend((i, p) for p in pos_here)
        if neg_here:
            by_name = sorted(
                neg_here, key=lambda p: _set_jaccard(s1_core[i], ref_core[p]), reverse=True
            )[:top_k_neg]
            by_locality = sorted(
                neg_here, key=lambda p: _set_jaccard(s1_loc[i], ref_loc[p]), reverse=True
            )[:top_k_neg]
            chosen = set(by_name) | set(by_locality)
            neg_pairs.extend((i, p) for p in chosen)

    rng = np.random.default_rng(seed)
    max_neg = int(len(pos_pairs) * neg_pos_ratio)
    if max_neg > 0 and len(neg_pairs) > max_neg:
        keep_idx = rng.choice(len(neg_pairs), size=max_neg, replace=False)
        neg_pairs = [neg_pairs[k] for k in keep_idx]

    s1_idx = np.array([p[0] for p in pos_pairs] + [p[0] for p in neg_pairs], dtype=np.int64)
    ref_idx = np.array([p[1] for p in pos_pairs] + [p[1] for p in neg_pairs], dtype=np.int64)
    labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int8)

    print(
        f"[train] built {len(pos_pairs)} positive / {len(neg_pairs)} negative "
        f"training pairs (ratio {len(neg_pairs) / max(len(pos_pairs), 1):.2f}:1)"
    )
    return s1_idx, ref_idx, labels


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    seed: int = 42,
    params: dict = None,
):
    """Train LightGBM with an internal validation split for early stopping,
    then calibrate on that same held-out split with isotonic regression.

    Returns (booster, calibrator, val_idx) so the caller can also compute
    threshold-tuning metrics on the exact same held-out rows.
    """
    params = dict(DEFAULT_LGB_PARAMS if params is None else params)
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = lgb.Dataset(X[train_idx], label=y[train_idx])
    val_set = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_set)

    n_estimators = params.pop("n_estimators", 500)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)],
    )

    raw_val_scores = booster.predict(X[val_idx], num_iteration=booster.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_val_scores, y[val_idx])

    gc.collect()
    return booster, calibrator, val_idx


def predict_calibrated(booster, calibrator, X: np.ndarray) -> np.ndarray:
    raw = booster.predict(X, num_iteration=booster.best_iteration)
    return calibrator.predict(raw)
