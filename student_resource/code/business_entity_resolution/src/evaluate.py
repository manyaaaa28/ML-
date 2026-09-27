"""
Macro-averaged F_0.5 exactly as the challenge scores it: per Source-1
entity, then averaged across all entities (singletons included - correct
empty prediction = 1.0, any false match on a true singleton = 0.0).
"""

from __future__ import annotations

from typing import Dict, Set


def f_beta_per_entity(true_ids: Set[str], pred_ids: Set[str], beta: float = 0.5) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(
    gt_map: Dict[str, Set[str]],
    pred_map: Dict[str, Set[str]],
    beta: float = 0.5,
) -> float:
    """Macro-average over every key in ``gt_map`` (all required S1 entities).

    A source1 id missing from ``pred_map`` is treated as an empty prediction
    (matches the "leave empty for no match" output contract).
    """
    if not gt_map:
        return 0.0
    total = 0.0
    for sid, true_ids in gt_map.items():
        pred_ids = pred_map.get(sid, set())
        total += f_beta_per_entity(true_ids, pred_ids, beta=beta)
    return total / len(gt_map)


def precision_recall_at(
    gt_map: Dict[str, Set[str]], pred_map: Dict[str, Set[str]]
):
    """Micro precision/recall across all matched-pair decisions (diagnostic only;
    the scored metric is the macro per-entity F0.5 above)."""
    tp = fp = fn = 0
    for sid, true_ids in gt_map.items():
        pred_ids = pred_map.get(sid, set())
        tp += len(true_ids & pred_ids)
        fp += len(pred_ids - true_ids)
        fn += len(true_ids - pred_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall
