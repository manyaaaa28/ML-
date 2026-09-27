"""
Thresholding, conflict resolution and the singleton confidence-band
safeguard - turns calibrated pairwise probabilities into the final
S1 -> {matched ref ids} mapping.

Conflict-resolution model (see README for the write-up of why this
replaces the connected-component + linear_sum_assignment approach
originally sketched in the plan):

    A Source 1 entity may legitimately match MANY Source 2/3 records (the
    ground truth has entities with 3-6+ matches). A Source 2/3 record,
    being a single real-world listing, should end up claimed by AT MOST
    ONE Source 1 entity. So the capacity constraint is one-sided: ref-side
    capacity 1, S1-side capacity unbounded.

    Under that capacity structure, the assignment problem decomposes
    exactly: for each ref record with multiple surviving S1 claimants,
    keeping the highest-probability claim is optimal - there is no coupling
    between different ref records' decisions, so no graph/component
    reasoning or Kuhn-Munkres assignment is needed to reach the exact
    optimum. This is also far cheaper: O(n log n) over the whole edge list
    instead of O(|V|^3) per connected component, so it scales cleanly to
    the full ~1.7M-entity test set.
"""

from __future__ import annotations

from typing import Dict, Set

import numpy as np
import pandas as pd

import evaluate as ev


def resolve_conflicts(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Keep, per ``ref_id``, only the highest-``prob`` surviving edge.

    ``pairs_df`` must have columns: source1_entity_id, ref_id, prob (plus
    whatever else the caller wants to carry through).
    """
    if pairs_df.empty:
        return pairs_df
    idx = pairs_df.groupby("ref_id")["prob"].idxmax()
    return pairs_df.loc[idx].reset_index(drop=True)


def apply_confidence_bands(
    pairs_df: pd.DataFrame,
    tau_global: float,
    tau_singleton: float,
) -> pd.DataFrame:
    """Filter pairs by the acceptance threshold, with a stricter bar in the
    borderline confidence band [tau_global, tau_singleton].

    Below tau_global: rejected outright.
    In [tau_global, tau_singleton]: accepted only with strict physical
    verification (street number AND postal token both match) - this is the
    band where a false accept is most likely to turn a true singleton into
    a costly false merge (macro F0.5 drops that entity from 1.0 to 0.0), so
    we demand corroborating physical evidence, not just name/string
    similarity, before trusting it.
    Above tau_singleton: accepted unconditionally (high-confidence).

    ``pairs_df`` needs columns: prob, street_number_match, postal_exact_match.
    """
    if pairs_df.empty:
        return pairs_df
    assert tau_singleton >= tau_global, "tau_singleton must be >= tau_global"

    above_all = pairs_df["prob"] > tau_singleton
    borderline = (pairs_df["prob"] >= tau_global) & (pairs_df["prob"] <= tau_singleton)
    verified = borderline & (pairs_df["street_number_match"] > 0) & (pairs_df["postal_exact_match"] > 0)

    keep_mask = above_all | verified
    return pairs_df[keep_mask].reset_index(drop=True)


def build_predictions(
    pairs_df: pd.DataFrame, required_s1_ids
) -> Dict[str, Set[str]]:
    """S1 entity -> set(matched ref ids). Every id in ``required_s1_ids``
    is present (empty set if no surviving candidate) - the "force empty
    prediction" rule."""
    pred: Dict[str, Set[str]] = {sid: set() for sid in required_s1_ids}
    if pairs_df.empty:
        return pred
    for sid, ref_id in zip(pairs_df["source1_entity_id"], pairs_df["ref_id"]):
        pred.setdefault(sid, set()).add(ref_id)
    return pred


def tune_thresholds(
    val_pairs_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    required_s1_ids,
    tau_global_grid=None,
    band_width_grid=None,
) -> Dict:
    """Grid-search (tau_global, tau_singleton) directly against macro F0.5
    on a held-out validation split. Returns the best combination and score.
    """
    if tau_global_grid is None:
        tau_global_grid = np.round(np.arange(0.40, 0.96, 0.05), 2)
    if band_width_grid is None:
        band_width_grid = [0.0, 0.05, 0.10, 0.15]

    best = {"tau_global": 0.5, "tau_singleton": 0.5, "f05": -1.0}
    for tau_g in tau_global_grid:
        for band in band_width_grid:
            tau_s = min(1.0, tau_g + band)
            filtered = apply_confidence_bands(val_pairs_df, tau_g, tau_s)
            resolved = resolve_conflicts(filtered)
            pred = build_predictions(resolved, required_s1_ids)
            score = ev.macro_f_beta(gt_map, pred, beta=0.5)
            if score > best["f05"]:
                best = {"tau_global": float(tau_g), "tau_singleton": float(tau_s), "f05": float(score)}
    return best
