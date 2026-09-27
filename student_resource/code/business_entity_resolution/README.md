# Business Entity Resolution Pipeline

Reproduction steps and an overview of the code. See the plan/architecture
doc (shared separately) for the full design rationale, and
`Documentation_template.md` (repo root) for the methodology write-up.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on Linux/macOS
pip install -r code/business_entity_resolution/requirements.txt
```

Tested with Python 3.12 on Windows (CPU-only, no GPU required/used).

## Layout

```
src/
  io_utils.py      # TSV/Parquet I/O, chunked readers
  preprocess.py     # name/address/country normalization, legal-suffix mining
  blocking.py        # 4-way candidate generation ensemble
  features.py         # pairwise similarity feature computation
  train.py             # hard-negative mining, LightGBM training, calibration
  match.py            # thresholding, singleton safeguard, conflict resolution
  evaluate.py          # macro-averaged F0.5 (matches the challenge's own formula)
  pipeline.py          # orchestration CLI (`dev` and `full` subcommands)
  sample.py            # builds a stratified dev sample from the full train data
```

## Reproducing the dev-sample run (recommended first step)

1. Build a sample (once; ~90s, reads the full train files but keeps a small subset):

   ```bash
   cd src
   python sample.py --train-dir ../../../dataset/train --out-dir ../../../dataset/sample \
       --n-s1 20000 --distractor-multiplier 8
   ```

2. Run the pipeline end-to-end on the sample (preprocess -> block -> feature ->
   train -> tune thresholds -> predict -> write outputs), reporting its own
   macro F0.5 on a held-out slice of the sample:

   ```bash
   python pipeline.py dev --sample-dir ../../../dataset/sample --output-dir ../../../output
   ```

3. Validate the output format:

   ```bash
   cd ../../..   # back to student_resource/
   python utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/sample
   ```

   (Point `--test-dir` at `dataset/sample` for the dev run since that's
   where `dev`'s held-out S1/S2/S3 files live; point it at `dataset/test`
   once running in `full` mode.)

## Full-scale run

`python pipeline.py full --train-dir dataset/train --test-dir dataset/test
--output-dir output` is the intended entry point for the real submission,
but **is not yet wired up** - see the `NotImplementedError` in
`pipeline.py`. The blocking stage as written fits a single TF-IDF matrix
over the whole reference pool (S2+S3), which is fine at dev-sample scale
(~390K rows) but won't fit resident in 8GB free RAM at full scale
(~10M S2+S3 rows). Full-scale needs the reference pool sharded (e.g. by
country, or by name first-character bucket) before each shard's TF-IDF
matrix is built - the rest of the pipeline (preprocess, features, train,
match) is scale-agnostic and reusable as-is once that shard loop exists.

## Key design decisions (see the plan doc for the full rationale)

- **No neural embedding feature.** CPU-only hardware with ~8GB free RAM
  made torch/sentence-transformers a real OOM/runtime risk; character
  n-gram TF-IDF + `anyascii` Unicode folding is the generalization hedge
  for unseen scripts/countries (France) instead.
- **Data-driven legal-suffix mining** (`preprocess.mine_legal_suffixes`) -
  no hardcoded suffix list, so it picks up French `SARL`/`SAS`/`EURL` from
  the data the same way it picks up US/India suffixes.
- **Locality is a token bag, not a single string** - addresses have
  reordered components (state-first, city-first, etc.), so
  `preprocess.extract_locality` returns the union of non-street tokens for
  Jaccard-style comparison rather than picking one segment positionally.
- **Postal-code matching is a weak, secondary feature**, not a blocking
  key - EDA on the real data showed this dataset's `business_address`
  field almost never contains an actual postal/PIN/ZIP code; the extracted
  "postal" token is usually just a house/plot number. See the comment in
  `preprocess.py` above `_POSTAL_RE`.
- **Conflict resolution is a per-ref-record max, not a bipartite
  assignment.** A Source 1 entity may have many true matches, but a
  Source 2/3 record should be claimed by at most one Source 1 entity - a
  one-sided capacity constraint that decomposes exactly into "keep the
  highest-probability claim per ref record," with no need for
  connected-component/`linear_sum_assignment` machinery. See the docstring
  at the top of `match.py`.
