"""
Build a small stratified dev sample from the full training data.

Used to iterate the pipeline in minutes instead of hours. The sample keeps
every ground-truth match for the sampled S1 entities (so recall is
measurable) plus a pool of random "distractor" S2/S3 records so blocking
actually has non-trivial work to do (recall isn't trivially 100% with no
noise to filter).

Run directly:
    python sample.py --train-dir ../../dataset/train --out-dir ../../dataset/sample \
        --n-s1 30000 --distractor-multiplier 8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import io_utils  # noqa: E402


def build_sample(
    train_dir: str,
    out_dir: str,
    n_s1: int = 30_000,
    distractor_multiplier: float = 8.0,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"[sample] loading {s1_path} ...")
    s1_full = io_utils.read_source_tsv(s1_path)
    n_s1 = min(n_s1, len(s1_full))
    s1_sample = s1_full.sample(n=n_s1, random_state=seed).reset_index(drop=True)
    print(f"[sample] S1: {len(s1_full)} -> {len(s1_sample)} sampled")
    del s1_full
    io_utils.collect()

    print(f"[sample] loading {gt_path} ...")
    gt_full = io_utils.read_ground_truth(gt_path)
    sampled_ids = set(s1_sample["entity_id"])
    gt_sample = gt_full[gt_full["source1_entity_id"].isin(sampled_ids)].reset_index(drop=True)
    print(f"[sample] ground truth: {len(gt_full)} -> {len(gt_sample)} rows")
    del gt_full
    io_utils.collect()

    matched_ids: set = set()
    for ids_str in gt_sample["matched_entity_ids"]:
        if ids_str:
            matched_ids.update(ids_str.split(","))
    print(f"[sample] {len(matched_ids)} distinct matched S2/S3 ids to keep")

    n_matched_s2 = sum(1 for i in matched_ids if i.startswith("S2-"))
    n_matched_s3 = sum(1 for i in matched_ids if i.startswith("S3-"))

    def sample_source(path: str, label: str, n_matched: int) -> pd.DataFrame:
        target_distractors = int(n_s1 * distractor_multiplier)
        kept_chunks = []
        total_rows = 0
        distractor_count = 0
        # First pass: figure out an approximate keep-probability for distractors
        # by streaming once; we don't know total row count up front cheaply,
        # so use a fixed generous probability and cap distractors afterward.
        approx_total = 5_100_000  # both S2/S3 are ~5M rows; good enough for a keep-prob estimate
        keep_prob = min(1.0, (target_distractors * 1.5) / approx_total)
        for chunk in io_utils.iter_source_tsv_chunks(path, chunksize=200_000):
            total_rows += len(chunk)
            is_match = chunk["entity_id"].isin(matched_ids)
            mask = np.asarray(is_match)
            if keep_prob > 0:
                rand_mask = rng.random(len(chunk)) < keep_prob
                mask = mask | rand_mask
            kept = chunk[mask]
            kept_chunks.append(kept)
        out = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else chunk.iloc[0:0]
        out = out.drop_duplicates(subset="entity_id")
        # Trim distractor surplus (keep all true matches, cap distractors).
        is_true_match = out["entity_id"].isin(matched_ids)
        true_rows = out[is_true_match]
        distractor_rows = out[~is_true_match]
        if len(distractor_rows) > target_distractors:
            distractor_rows = distractor_rows.sample(n=target_distractors, random_state=seed)
        out = pd.concat([true_rows, distractor_rows], ignore_index=True)
        print(
            f"[sample] {label}: streamed {total_rows} rows -> kept {len(out)} "
            f"({len(true_rows)} true matches + {len(distractor_rows)} distractors)"
        )
        return out

    s2_sample = sample_source(s2_path, "S2", n_matched_s2)
    io_utils.collect()
    s3_sample = sample_source(s3_path, "S3", n_matched_s3)
    io_utils.collect()

    os.makedirs(out_dir, exist_ok=True)
    s1_sample.to_csv(os.path.join(out_dir, "sample_source1.tsv"), sep="\t", index=False)
    s2_sample.to_csv(os.path.join(out_dir, "sample_source2.tsv"), sep="\t", index=False)
    s3_sample.to_csv(os.path.join(out_dir, "sample_source3.tsv"), sep="\t", index=False)
    gt_sample.to_csv(os.path.join(out_dir, "sample_ground_truth.tsv"), sep="\t", index=False)
    print(f"[sample] wrote sample files to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="../../dataset/train")
    ap.add_argument("--out-dir", default="../../dataset/sample")
    ap.add_argument("--n-s1", type=int, default=30_000)
    ap.add_argument("--distractor-multiplier", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_sample(args.train_dir, args.out_dir, args.n_s1, args.distractor_multiplier, args.seed)


if __name__ == "__main__":
    main()
