"""
End-to-end orchestration: preprocess -> block -> feature -> train -> match -> output.

Two entry points share every stage's code, so validating on a sample and
running at full scale never diverge into separate implementations:

    python pipeline.py dev   --sample-dir ../../../dataset/sample --output-dir ../../../output
    python pipeline.py full  --train-dir ../../../dataset/train --test-dir ../../../dataset/test \
                              --output-dir ../../../output

``dev`` carves its own held-out validation split out of the (labeled)
sample so it can tune thresholds and report macro F0.5 against real ground
truth. ``full`` does the same against a slice of the (labeled) full train
set, then applies the frozen model/thresholds to the (unlabeled) real test
set for the submission files.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
import io_utils
import preprocess as pp
import blocking as bl
import features as feat_mod
import train as train_mod
import match as match_mod
import evaluate as ev

FEATURE_TFIDF_COL = feat_mod.FEATURE_NAMES.index("name_tfidf_cosine")
FEATURE_STREET_COL = feat_mod.FEATURE_NAMES.index("street_number_match")
FEATURE_POSTAL_COL = feat_mod.FEATURE_NAMES.index("postal_exact_match")

BLOCKING_PARAMS = dict(
    tfidf_ngram_range=(3, 5),
    tfidf_max_features=200_000,
    tfidf_top_n=20,
    tfidf_threshold=0.25,
    token_max_postings=8000,
    addr_max_postings=8000,
    sn_window=10,
    # Per-entity caps on strategies A/D (see blocking.py's module docstring
    # and token_index_candidates' docstring): without these, candidate sets
    # can run into the thousands per entity, which makes the downstream
    # feature+model stage infeasible at real scale (millions of S1 entities
    # x thousands of candidates each). 100 was chosen empirically on the
    # dev sample: recall 98.7% (mean candidate volume ~187/entity) vs.
    # 99.99% uncapped (~5,150/entity) - a ~27x volume cut for a 1.3pp
    # recall cost, comfortably above the 95% target. See the plan doc /
    # README for the full recall-vs-volume sweep (100/300/500/uncapped).
    token_per_entity_cap=100,
    addr_per_entity_cap=100,
)


def log(msg: str, t0: float) -> None:
    print(f"[pipeline +{time.time() - t0:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Shared stages
# --------------------------------------------------------------------------

def mine_suffixes(*name_series) -> set:
    tokens = []
    for s in name_series:
        tokens.extend(pp.tokenize(pp.clean_name(n)) for n in s)
    return pp.mine_legal_suffixes(tokens, min_count=200, min_trailing_ratio=0.5)


def normalize(df: pd.DataFrame, suffix_vocab: set) -> pd.DataFrame:
    return pp.normalize_frame(df, suffix_vocab=suffix_vocab)


class BlockingIndices:
    def __init__(self, ref_norm: pd.DataFrame, params: dict = BLOCKING_PARAMS):
        self.params = params
        self.token_index = bl.build_token_index(ref_norm["name_core_tokens"].tolist())
        self.vectorizer = bl.fit_char_ngram_vectorizer(
            ref_norm["name_norm"].tolist(),
            ngram_range=params["tfidf_ngram_range"],
            max_features=params["tfidf_max_features"],
        )
        self.ref_tfidf = self.vectorizer.transform(ref_norm["name_norm"].tolist()).astype(np.float32)
        self.sn_index = bl.build_sorted_neighborhood_index(
            ref_norm["locality"].tolist(), ref_norm["name_norm"].tolist()
        )
        self.addr_index = bl.build_address_token_index(ref_norm["addr_norm"].tolist())


def run_blocking(s1_norm: pd.DataFrame, idxs: BlockingIndices) -> List[Set[int]]:
    p = idxs.params
    cand_a = bl.token_index_candidates(
        s1_norm["name_core_tokens"].tolist(), idxs.token_index, p["token_max_postings"],
        per_entity_cap=p["token_per_entity_cap"],
    )
    # Uses the ref TF-IDF matrix already fitted+transformed once in
    # BlockingIndices instead of blocking.tfidf_topk_candidates (which would
    # re-transform the reference corpus on every call).
    cand_b = _tfidf_candidates_precomputed(s1_norm, idxs, p)
    cand_c = bl.sorted_neighborhood_candidates(
        s1_norm["locality"].tolist(), s1_norm["name_norm"].tolist(), idxs.sn_index, p["sn_window"]
    )
    cand_d = bl.address_token_candidates(
        s1_norm["addr_norm"].tolist(), idxs.addr_index, p["addr_max_postings"],
        per_entity_cap=p["addr_per_entity_cap"],
    )
    return bl.union_candidate_sets(cand_a, cand_b, cand_c, cand_d)


def _tfidf_candidates_precomputed(s1_norm, idxs: BlockingIndices, p: dict, chunk_size: int = 200_000):
    from sparse_dot_topn import sp_matmul_topn

    s1_texts = s1_norm["name_norm"].tolist()
    ref_mat_t = idxs.ref_tfidf.T.tocsr()
    out: List[Set[int]] = [set() for _ in range(len(s1_texts))]
    for start in range(0, len(s1_texts), chunk_size):
        end = min(start + chunk_size, len(s1_texts))
        chunk_mat = idxs.vectorizer.transform(s1_texts[start:end]).astype(np.float32)
        sim = sp_matmul_topn(chunk_mat, ref_mat_t, top_n=p["tfidf_top_n"], threshold=p["tfidf_threshold"], sort=False).tocsr()
        for local_i in range(sim.shape[0]):
            row = sim.getrow(local_i)
            if row.nnz:
                out[start + local_i] = set(row.indices.tolist())
        del chunk_mat, sim
    gc.collect()
    return out


def explode_candidates(s1_norm: pd.DataFrame, ref_norm: pd.DataFrame, candidate_positions: List[Set[int]]) -> pd.DataFrame:
    s1_ids = s1_norm["entity_id"].values
    ref_ids = ref_norm["entity_id"].values
    s1_pos_list, ref_pos_list = [], []
    for i, positions in enumerate(candidate_positions):
        for p in positions:
            s1_pos_list.append(i)
            ref_pos_list.append(p)
    if not s1_pos_list:
        return pd.DataFrame(columns=["s1_pos", "ref_pos", "source1_entity_id", "ref_id"])
    s1_pos_arr = np.array(s1_pos_list, dtype=np.int64)
    ref_pos_arr = np.array(ref_pos_list, dtype=np.int64)
    return pd.DataFrame(
        {
            "s1_pos": s1_pos_arr,
            "ref_pos": ref_pos_arr,
            "source1_entity_id": s1_ids[s1_pos_arr],
            "ref_id": ref_ids[ref_pos_arr],
        }
    )


def write_candidate_tsv(s1_norm: pd.DataFrame, candidate_positions: List[Set[int]], ref_ids: np.ndarray, path: str) -> None:
    s1_ids = s1_norm["entity_id"].values
    rows = []
    for i, sid in enumerate(s1_ids):
        ids = sorted(ref_ids[list(candidate_positions[i])]) if candidate_positions[i] else []
        rows.append((sid, ",".join(ids)))
    out = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    io_utils.write_id_list_tsv(out, path, "source1_entity_id", "candidate_entity_ids")


def compute_features_for_pairs(
    s1_norm: pd.DataFrame, ref_norm: pd.DataFrame, idxs: BlockingIndices, pairs_df: pd.DataFrame, batch_size: int = 300_000
) -> np.ndarray:
    if pairs_df.empty:
        return np.zeros((0, len(feat_mod.FEATURE_NAMES)), dtype=np.float32)
    s1_tfidf = idxs.vectorizer.transform(s1_norm["name_norm"].tolist()).astype(np.float32)
    # Precompute per-row caches ONCE for this frame pair, reused across every
    # chunk below - see features.py's module docstring for why this matters.
    s1_cache = feat_mod.precompute_frame_cache(s1_norm)
    ref_cache = feat_mod.precompute_frame_cache(ref_norm)
    feats_chunks = []
    n = len(pairs_df)
    s1_pos = pairs_df["s1_pos"].values
    ref_pos = pairs_df["ref_pos"].values
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s1_idx_b = s1_pos[start:end]
        ref_idx_b = ref_pos[start:end]
        a = s1_tfidf[s1_idx_b]
        b = idxs.ref_tfidf[ref_idx_b]
        cos = np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)
        f = feat_mod.compute_pair_features_batch(s1_idx_b, ref_idx_b, s1_cache, ref_cache, tfidf_cosine=cos)
        feats_chunks.append(f)
    return np.concatenate(feats_chunks, axis=0)


# --------------------------------------------------------------------------
# Dev (sample) pipeline
# --------------------------------------------------------------------------

def run_dev(sample_dir: str, output_dir: str, val_frac: float = 0.3, seed: int = 42) -> None:
    t0 = time.time()
    s1_all = io_utils.read_source_tsv(os.path.join(sample_dir, "sample_source1.tsv"))
    s2 = io_utils.read_source_tsv(os.path.join(sample_dir, "sample_source2.tsv"))
    s3 = io_utils.read_source_tsv(os.path.join(sample_dir, "sample_source3.tsv"))
    gt_all = io_utils.read_ground_truth(os.path.join(sample_dir, "sample_ground_truth.tsv"))
    log(f"loaded S1={len(s1_all)} S2={len(s2)} S3={len(s3)} GT={len(gt_all)}", t0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(s1_all))
    n_val = int(len(s1_all) * val_frac)
    val_rows, train_rows = perm[:n_val], perm[n_val:]
    s1_train = s1_all.iloc[train_rows].reset_index(drop=True)
    s1_val = s1_all.iloc[val_rows].reset_index(drop=True)
    log(f"split: train S1={len(s1_train)} val S1={len(s1_val)}", t0)

    suffixes = mine_suffixes(s1_all["business_name"], s2["business_name"], s3["business_name"])
    log(f"mined {len(suffixes)} legal-suffix tokens", t0)

    s1_train_n = normalize(s1_train, suffixes)
    s1_val_n = normalize(s1_val, suffixes)
    s2_n = normalize(s2, suffixes)
    s3_n = normalize(s3, suffixes)
    ref_n = pd.concat([s2_n, s3_n], ignore_index=True)
    ref_ids = ref_n["entity_id"].values
    log(f"normalized. ref pool size={len(ref_n)}", t0)

    gt_map = {}
    for sid, ids in zip(gt_all["source1_entity_id"], gt_all["matched_entity_ids"]):
        gt_map[sid] = set(ids.split(",")) if ids else set()
    gt_map_train = {sid: gt_map.get(sid, set()) for sid in s1_train_n["entity_id"]}
    gt_map_val = {sid: gt_map.get(sid, set()) for sid in s1_val_n["entity_id"]}

    idxs = BlockingIndices(ref_n)
    log("fitted blocking indices", t0)

    cand_train = run_blocking(s1_train_n, idxs)
    cand_val = run_blocking(s1_val_n, idxs)
    log("blocking done for train/val", t0)

    # blocking quality report (on val, the honest held-out check)
    found = total_true = 0
    for i, sid in enumerate(s1_val_n["entity_id"]):
        true_ids = gt_map_val.get(sid, set())
        if not true_ids:
            continue
        cand_ids = set(ref_ids[list(cand_val[i])]) if cand_val[i] else set()
        found += len(true_ids & cand_ids)
        total_true += len(true_ids)
    pair_recall = found / total_true if total_true else float("nan")
    sizes = [len(c) for c in cand_val]
    log(f"VAL blocking pair recall = {pair_recall:.4f}  mean cand/entity = {np.mean(sizes):.1f}", t0)

    val_pairs_df = explode_candidates(s1_val_n, ref_n, cand_val)

    s1_idx, ref_idx, labels = train_mod.build_training_pairs(
        s1_train_n, ref_n, cand_train, gt_map_train, top_k_neg=5, neg_pos_ratio=6.0, seed=seed
    )
    train_pairs_for_features = pd.DataFrame({"s1_pos": s1_idx, "ref_pos": ref_idx})
    X_train = compute_features_for_pairs(s1_train_n, ref_n, idxs, train_pairs_for_features)
    y_train = labels
    log(f"training feature matrix: {X_train.shape}", t0)

    booster, calibrator, _ = train_mod.train_classifier(X_train, y_train, seed=seed)
    log("model trained + calibrated", t0)

    X_val = compute_features_for_pairs(s1_val_n, ref_n, idxs, val_pairs_df)
    if len(val_pairs_df):
        val_probs = train_mod.predict_calibrated(booster, calibrator, X_val)
        val_pairs_df = val_pairs_df.copy()
        val_pairs_df["prob"] = val_probs
        val_pairs_df["street_number_match"] = X_val[:, FEATURE_STREET_COL]
        val_pairs_df["postal_exact_match"] = X_val[:, FEATURE_POSTAL_COL]
    else:
        val_pairs_df["prob"] = []
        val_pairs_df["street_number_match"] = []
        val_pairs_df["postal_exact_match"] = []
    log("scored validation candidates", t0)

    required_val_ids = list(s1_val_n["entity_id"])
    best = match_mod.tune_thresholds(val_pairs_df, gt_map_val, required_val_ids)
    log(f"tuned thresholds: {best}", t0)

    filtered = match_mod.apply_confidence_bands(val_pairs_df, best["tau_global"], best["tau_singleton"])
    resolved = match_mod.resolve_conflicts(filtered)
    val_pred = match_mod.build_predictions(resolved, required_val_ids)
    macro_f05 = ev.macro_f_beta(gt_map_val, val_pred, beta=0.5)
    precision, recall = ev.precision_recall_at(gt_map_val, val_pred)
    log(f"VAL macro F0.5 = {macro_f05:.4f}  micro precision={precision:.4f} recall={recall:.4f}", t0)

    os.makedirs(output_dir, exist_ok=True)
    write_candidate_tsv(s1_val_n, cand_val, ref_ids, os.path.join(output_dir, "candidate_pairs.tsv"))
    pred_rows = [(sid, ",".join(sorted(val_pred.get(sid, set())))) for sid in required_val_ids]
    pred_df = pd.DataFrame(pred_rows, columns=["source1_entity_id", "matched_entity_ids"])
    io_utils.write_id_list_tsv(pred_df, os.path.join(output_dir, "matching_results.tsv"), "source1_entity_id", "matched_entity_ids")
    log(f"wrote output files to {output_dir}", t0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    dev = sub.add_parser("dev")
    dev.add_argument("--sample-dir", default="../../../dataset/sample")
    dev.add_argument("--output-dir", default="../../../output")
    dev.add_argument("--val-frac", type=float, default=0.3)
    dev.add_argument("--seed", type=int, default=42)

    full = sub.add_parser("full")
    full.add_argument("--train-dir", default="../../../dataset/train")
    full.add_argument("--test-dir", default="../../../dataset/test")
    full.add_argument("--output-dir", default="../../../output")
    full.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    if args.mode == "dev":
        run_dev(args.sample_dir, args.output_dir, args.val_frac, args.seed)
    elif args.mode == "full":
        raise NotImplementedError(
            "full-scale mode is not wired up yet - it needs sharded blocking "
            "(the reference TF-IDF matrix for ~10M S2+S3 rows won't fit resident "
            "in 8GB RAM as a single matrix) before it can run on the true "
            "train/test data. See the plan doc's Open Risks section."
        )


if __name__ == "__main__":
    main()
