# Business Entity Resolution Pipeline — Plan & Architecture

Sep 25, 2026 · @Swarnim

This document captures the plan and architecture for the ML Challenge 2026 Business Entity Resolution pipeline: given noisy business records from three sources, find every Source 2/Source 3 record matching each Source 1 entity, optimizing macro-averaged F0.5 (precision weighted 2x over recall).

**Scope decisions locked in:**

- **Classical-only pipeline** — no neural embedding model for now. Given CPU-only hardware with limited free RAM, we rely on character n-gram TF-IDF similarity plus a data-driven, script-agnostic transliteration/normalization layer as the generalization hedge for the unseen France country label, instead of installing torch/sentence-transformers. An embedding feature stays a clean future add-on, not a blocker.
- **Sample-first development** — build and validate the full pipeline against a small stratified sample (tens of thousands of entities, minutes not hours per iteration), confirm blocking recall and F0.5 on a held-out split, then run the *same unmodified code* once over the full-scale data.

## Constraints & Environment

**Hardware:** 12-core Intel i5-1235U laptop, 16GB RAM (\~8GB free), integrated Iris Xe graphics only — no discrete GPU.

**Data scale:**

|  | Source 1 | Source 2 | Source 3 | Ground truth |
| --- | --- | --- | --- | --- |
| Train | \~2.2M | \~5.0M | \~5.3M | \~2.2M rows |
| Test | \~1.73M | \~4.9M | \~5.1M | — |

\~26M rows total across 7 files, \~2.4GB on disk.

**Installed packages (venv):** pandas, numpy, scipy, scikit-learn, lightgbm, rapidfuzz, anyascii, pyarrow, sparse\_dot\_topn, psutil. No torch/sentence-transformers installed (deliberately, per the embeddings scope decision below).

**Scope decisions:**

1. No neural embedding similarity feature for now — avoids multi-hour CPU inference and OOM risk on 8GB free RAM. `anyascii`-based Unicode folding covers script-agnostic name normalization instead.
2. Develop against a stratified sample first, validate blocking recall and F0.5, then run the identical pipeline code once at full scale as a background job.

These constraints drive every downstream design choice: chunked/streaming I/O over the 5M-row S2/S3 files, Parquet caching of normalized fields (avoid re-parsing TSV repeatedly), batched feature/inference processing (200–500K pairs at a time), and `sparse_dot_topn` for bounded-memory sparse top-k similarity instead of a dense nearest-neighbor search.

## Repo Layout

```
student_resource/
  code/business_entity_resolution/
    src/
      preprocess.py     # name/address/country normalization
      blocking.py        # candidate generation, union of strategies
      features.py        # pairwise feature computation
      train.py            # builds training pairs, trains LightGBM
      match.py           # scoring, thresholding, bipartite assignment
      pipeline.py         # CLI orchestration: --mode sample|full, stage flags
      evaluate.py         # F0.5 macro-average on held-out split
      io_utils.py          # chunked TSV readers/writers, dtype-light loading
    README.md
    requirements.txt
  output/
    matching_results.tsv
    candidate_pairs.tsv
  Documentation_template.md   (filled in at the end)
```

`pipeline.py` exposes a `--mode {sample,full}` flag and a `--sample-size` parameter so the *exact same code path* runs in development and in the final full-scale run — there is no separate throwaway "toy" script to diverge from the real pipeline.

## Architecture Diagram

```
                         ┌──────────────────────────────────────────────────────────┐
                         │                RAW INPUT                     │
                         │  train_source{1,2,3}.tsv, train_ground_truth │
                         │  test_source{1,2,3}.tsv                      │
                         └──────────────────────┬────────────────────────────────────────────┐
                                              ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  STAGE 1 — PREPROCESS  (preprocess.py)                                │
   │  per source, per row, one pass:                                       │
   │    name  → normalize() → {name_norm, name_tokens, name_core_set}      │
   │    addr  → normalize() → {addr_norm, postal_token, locality_token}    │
   │    country → passthrough + normalization-table lookup (generic fallback)│
   │  persisted to Parquet (io_utils.py) — never re-parsed from TSV again  │
   └────────────────────────────────────┬───────────────────────────────────────────────┐
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  STAGE 2 — BLOCKING  (blocking.py)      "recall stage"                │
   │                                                                        │
   │   S1 normalized ──┬─► [A] token-inverted-index lookup ─┐              │
   │                    ├─► [B] TF-IDF char-ngram top-k     ├─► UNION per  │
   │   S2/S3 normalized ┤     (sparse_dot_topn)              │   S1 entity │
   │                    └─► [C] sorted-neighborhood on       ┘              │
   │                          postal_token / locality_token                │
   │                                                                        │
   │   measured on held-out split: recall ≥ 95%, reduction ratio           │
   │   → candidate_pairs.tsv  (S1_id, [S2/S3 candidate ids])               │
   └─────────────────────────────────────┬───────────────────────────────────────────────┐
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  STAGE 3 — PAIRWISE FEATURES  (features.py)   "precision stage input" │
   │  for every (S1, candidate) pair surviving blocking:                   │
   │    name:  levenshtein, jaro-winkler, token jaccard/dice,              │
   │           tfidf cosine (reuse Stage-2 vectors), LCS ratio,            │
   │           suffix-normalized exact flag                                │
   │    addr:  token jaccard, postal exact flag, locality match,           │
   │           edit distance, street-number match flag                    │
   │    meta:  country match flag, source pair (S1–S2 vs S1–S3),           │
   │           name length ratio, shared-token count                      │
   │  → feature matrix, processed in batches (200–500K pairs/batch)        │
   └─────────────────────────────────────┬───────────────────────────────────────────────┐
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  STAGE 4 — PAIRWISE CLASSIFIER  (train.py)                            │
   │    positives = ground truth pairs                                     │
   │    negatives = hard negatives (every other blocked candidate for      │
   │                that S1, not in ground truth)                         │
   │    model = LightGBM binary classifier + isotonic calibration          │
   │  → calibrated match_probability per (S1, candidate) pair              │
   └─────────────────────────────────────┬───────────────────────────────────────────────┐
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  STAGE 5 — THRESHOLD + GLOBAL CONSISTENCY  (match.py)                 │
   │    threshold τ tuned against macro F0.5 on held-out split             │
   │    (not 0.5 — expect high, precision weighted 2×)                    │
   │    conflict resolution: greedy highest-score-first per S2/S3 record   │
   │    (exact assignment via linear_sum_assignment only within small      │
   │     connected components, if data shows it's needed)                 │
   │    no candidate clears τ → force empty prediction                     │
   └─────────────────────────────────────┬───────────────────────────────────────────────┐
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  STAGE 6 — OUTPUT + VALIDATE                                          │
   │    matching_results.tsv, candidate_pairs.tsv                          │
   │    utils/validate_submission.py  →  evaluate.py (macro F0.5)          │
   └──────────────────────────────────────────────────────────────────────────┘
```

## Stage 1 — Preprocessing

**Name normalization:**

- Lowercase, strip punctuation, collapse whitespace, expand `&` → `and`.
- Unicode NFKD normalize and ASCII-fold via `anyascii` (MIT-licensed, already installed) — folds Devanagari, Kannada, French-accented, and any other script into a comparable ASCII form without hardcoding per-language rules.
- **Data-driven legal-suffix dictionary**: mine frequent trailing tokens across all sources (train *and* test, since test adds France) by frequency + position, rather than a hand-typed list of `Corp/Ltd/Pvt/...`. This is what lets the suffix-stripping generalize to French forms (`SARL`, `SAS`) without special-casing them.
- Output both a full normalized string and a sorted core-token set (post-suffix-strip) for token-Jaccard blocking/features.

**Address normalization:**

- Small abbreviation map (Rd/Road, St/Street, Apt/Flat) with pass-through fallback for unmapped tokens — never fails on an unknown abbreviation.
- Generic postal-code-shaped token extracted via regex tuned to match alphanumeric formats broadly (not just `\d{5}`), so it doesn't silently fail on France's 5-digit codes or India's PIN codes.
- Locality guess = last non-empty comma-separated segment before the country token.
- No full geocoding/address-parsing service (prohibited by the challenge rules); an offline library like libpostal is optional and would be clearly documented as making no network/database calls if used — not planned for the initial build.

**Country handling:**

- Kept as a categorical field with country-specific normalization tables for confidently-detected US/India.
- **Generic fallback path** for every other label (France, or anything unseen) — must not fail or drop rows. This is enforced by construction: the normalization dispatch always has a default branch, never an explicit allowlist of `{US, India}`.

## Stage 2 — Blocking / Candidate Generation

Recall-first: combine multiple cheap strategies and take the **union** per Source 1 entity.

1. **Token-overlap blocking** — inverted index on significant name tokens (stopword/suffix-filtered).
2. **TF-IDF (char n-gram, 2–4 grams) cosine top-k** via `sparse_dot_topn` — typo/transliteration tolerant, and character n-grams are what generalize to unseen scripts/countries without hardcoding (fit per-country-bucket where feasible to shrink the matrix, generic-bucket fallback otherwise).
3. **Sorted-neighborhood / exact match** on normalized postal code, then on locality token.
4. Optional cheap country-match gate applied *before* the above as a pre-filter, never as the sole filter (addresses/country labels are noisy enough that a hard gate alone would lose recall).

datasketch MinHash/LSH is **not** included in the initial build (not installed, and TF-IDF top-k already covers typo tolerance reasonably at this scale) — flagged as an add-on if the recall check below comes up short, not a blocker.

**The final, narrowed candidate set that gets fed to the matching model is what goes into `candidate_pairs.tsv`** — if there are multiple filtering passes, only the last one counts, per the challenge's output format rules.

**Validation target:** measure blocking quality on a held-out validation split — pair completeness (recall) and reduction ratio. Target recall ≥ \~95% before moving to features/model; no downstream classifier can recover a match blocking never surfaced.

## Stage 3 — Pairwise Feature Engineering

Computed for every (Source 1, candidate) pair that survives blocking:

**Name features:** normalized Levenshtein / edit distance and Jaro-Winkler (via `rapidfuzz`, C-optimized), token Jaccard/Dice, TF-IDF cosine (reusing Stage-2 vectors where possible instead of recomputing), longest-common-substring ratio, suffix-normalized exact-match flag.

**Address features:** token Jaccard, postal-code exact-match flag, locality-token match, edit distance, street-number match flag.

**Meta features:** country-match flag, source pair indicator (S1–S2 vs S1–S3), name length ratio, shared-token count.

No embedding-similarity feature in this build (per the scope decision) — `features.py` is structured so an embedding column could be appended later without touching the rest of the pipeline.

Processed in batches of 200–500K candidate pairs at a time to stay within the memory budget — never materializes the full feature matrix for all candidate pairs at once.

## Stage 4 — Pairwise Match Classifier

**Training pairs:**

- Positives: ground-truth (S1, matched) pairs.
- Negatives: **hard negatives** — every other candidate blocking retrieved for that Source 1 entity that isn't a true match. Critical for a realistic precision/recall tradeoff, since it teaches the model the fine-grained distinction *within* what blocking already considered plausible — exactly the decision it faces at inference time. Random negatives from the whole dataset would be trivially distinguishable and teach too little.

**Model:** LightGBM binary classifier — open-source (MIT), tiny, well under the 8B-parameter cap, robust across heterogeneous similarity features, and gives feature importances for the methodology write-up. Chosen over XGBoost mainly for training speed on this CPU-bound machine (histogram-based, faster on millions of rows with limited cores/RAM).

**Calibration:** isotonic regression (via scikit-learn) on the raw model scores, so the resulting probabilities are meaningful for direct thresholding in Stage 5.

## Stage 5 — Thresholding & Global Consistency

**Threshold tuning:** the acceptance threshold τ is tuned directly against macro-averaged F0.5 on a held-out validation split — not a default 0.5 cutoff. Because F0.5 weights precision 2× over recall, the optimal cutoff is provably higher than 0.5 in general; tuning directly on the target metric (not accuracy/F1) is the only way to actually optimize what's being scored. Per-country thresholds are a plausible refinement to test afterward (France may have thinner training signal / calibration risk) but aren't committed to up front.

**Global consistency (as actually implemented - see "Implementation Results" below for why this changed from the plan above):** the real capacity structure here is one-sided - a Source 1 entity may legitimately have many true matches, but a Source 2/3 record should be claimed by at most one Source 1 entity. Under that structure the optimal resolution is exact and simple: for each ref record with multiple surviving claims, keep only the highest-probability one. This decomposes per-ref-record with no coupling between decisions, so it needs no graph/component reasoning or `linear_sum_assignment` at all - and it's O(n log n) over the whole edge list, which is what actually scales to the full test set. The connected-component + exact-assignment idea below (adopted from the external review) assumed a symmetric one-to-one capacity on both sides, which doesn't match this problem's many-matches-per-S1 ground truth.

**Singletons:** when no candidate clears τ for a Source 1 entity, the prediction is forced empty. A correct empty prediction on a true singleton scores 1.0; any false match on it scores 0.0 — so singleton precision matters as much as multi-match precision.

## Stage 6 — Output & Validation

**`matching_results.tsv`** — columns `source1_entity_id`, `matched_entity_ids`. One row per Source 1 test entity, empty string for singletons, comma-separated IDs with no quoting, no duplicate IDs within a list, no duplicate `source1_entity_id` rows, only S2-/S3- IDs that exist in the test set.

**`candidate_pairs.tsv`** — columns `source1_entity_id`, `candidate_entity_ids`. Same formatting rules. This is the exact candidate set fed to the final matching model at inference — every matched ID in `matching_results.tsv` must appear here (matches ⊆ candidates).

**Validation:** run `utils/validate_submission.py` (stdlib-only, provided by the challenge) locally before every leaderboard upload — it checks header format, duplicate rows/IDs, self-matches, wrong-prefix IDs, missing required S1 entities, and (optionally, with `--check-ids`) that IDs actually exist in the test set.

**`evaluate.py`** computes our own macro-averaged F0.5 on a held-out training split before trusting any leaderboard score, using the same per-entity formula the challenge uses: `F0.5 = (1.25 × P × R) / (0.25 × P + R)`, averaged across all Source 1 entities including singletons.

## Method Comparison: Chosen vs Alternatives

| Stage | Method chosen | Alternatives considered | Why this wins here |
| --- | --- | --- | --- |
| Blocking | Union of token-inverted-index + char-ngram TF-IDF top-k (`sparse_dot_topn`) + sorted-neighborhood on postal/locality | MinHash/LSH (datasketch), FAISS ANN, pure rule-based blocking keys only | Sorted-neighborhood alone breaks on missing/inconsistent postal data (a documented noise pattern); TF-IDF top-k is typo/transliteration-tolerant and `sparse_dot_topn` scales to millions of rows in bounded memory without a GPU. MinHash/LSH would add robustness for heavy typos but isn't installed and isn't clearly needed once TF-IDF top-k is in the union — add only if the sample-stage recall check comes up short. |
| Blocking similarity space | Character n-grams (2–4), not word n-grams | Word-level TF-IDF only | Character n-grams generalize to France/unseen scripts without hardcoding — the same mechanism handles diacritic folding, transliteration noise, and typos, whereas word-level blocking fails completely on a single character mismatch. |
| Matching model | LightGBM (GBDT) on hand-engineered pairwise features | XGBoost, logistic regression, small Siamese/transformer net | GBDT on interpretable distance/similarity features is the standard, well-proven approach for entity resolution at this scale — trains in minutes on CPU, gives feature importances for the write-up, trivially satisfies the license/8B-param constraint, and doesn't need embeddings to generalize. LightGBM over XGBoost mainly for training speed on this CPU-bound machine. A neural approach would need the embedding layer already ruled out for hardware reasons — without it, a net has no accuracy edge over GBDT here and costs more to train/tune. |
| Negative sampling | Hard negatives = all non-match blocking survivors | Random negatives from whole dataset | Random negatives are almost always trivially distinguishable (wrong country, totally different name) and teach the model too little — it needs to learn the fine-grained distinction *within* what blocking considered plausible, exactly the decision it faces at inference time. |
| Thresholding | τ tuned directly on macro F0.5, not 0.5 default | Fixed 0.5 cutoff, per-country thresholds | F0.5's 2× precision weighting means the optimal cutoff is provably higher than 0.5 in general; tuning directly on the target metric is the only way to actually optimize what's being scored. Per-country thresholds are a plausible refinement to test once the global one works (France may have thinner training signal) — flagged, not committed up front. |
| Conflict resolution | Greedy highest-score-first, with exact assignment as a fallback on small components | Global `linear_sum_assignment` everywhere, no conflict resolution at all | Skipping conflict resolution risks the same S2/S3 record being claimed by multiple S1 entities, directly hurting precision (and F0.5 harder than F1). Exact bipartite assignment is provably better but is O(n³)-ish per component and won't scale if any connected component (e.g. generic names like "Summit Inc") gets large — greedy is the safe default, exact assignment an opportunistic upgrade where cheap. |

**Bottom line:** blocking-ensemble → LightGBM pairwise classifier → F0.5-tuned threshold → greedy global assignment is the right architecture for this problem given the constraints (no embeddings, CPU-only, 8GB free RAM, open-set countries, precision-heavy metric). It's the standard, well-understood ER architecture, cheap enough to run twice (sample, then full), and every piece has a clear country-agnostic fallback so France doesn't need special-casing anywhere.

## Execution Sequence

1. **Sample draw:** a stratified sample of \~20–50K S1 entities across US/India, all their ground-truth matches, plus the S2/S3 rows blocking would retrieve for them — fast iteration loop, minutes not hours per run.
2. **Blocking iteration:** build preprocess → blocking, measure recall/reduction ratio, iterate blocking strategies until ≥ 95% recall on the sample's held-out split.
3. **Model iteration:** build features → train LightGBM → tune threshold → measure macro F0.5 on the sample's held-out split.
4. **Freeze + scale:** freeze the pipeline code, then run `pipeline.py --mode full` once, unattended (likely a multi-hour background run) over the true 2.2M/1.73M data — no code changes at this point, just resourcing (chunked I/O, batched inference, Parquet caching) so it survives on 8GB free RAM.
5. **Final validation:** validate full-scale output with `utils/validate_submission.py`, compute final F0.5 on a full held-out train split, fill in `Documentation_template.md`.

## Open Risks & Follow-ups

- ~~Connected-component size at full scale~~ — **resolved**: cap exact `linear_sum_assignment` bipartite matching to connected components with |V| ≤ 100 nodes; fall back to greedy highest-score-first for larger components (e.g. generic names like "Summit Inc" pulling in thousands of candidates). See *Adopted Refinements* below.
- **Embeddings deferred, not ruled out:** a small permissively-licensed multilingual sentence-embedding model (e.g. paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0) remains a clean future layer for France/unseen-script generalization once the classical pipeline is validated and if time/memory allow — `features.py` is structured to accept it without a redesign.
- **Per-country thresholds:** a possible refinement over a single global threshold, worth testing once the global F0.5-tuned threshold is working, especially if France shows weaker calibration due to thinner training signal.

## Adopted Refinements (External Strategy Review)

An external strategy-review pass proposed a set of concrete implementation refinements on top of this plan. Assessed against our actual constraints and evaluated on merit rather than adopted wholesale:

**Adopted outright** (genuine improvements, no scope change):

- **Asymmetric name features** — containment Jaccard and substring ratio, added to Stage 3's name-feature set to better handle DBA/trade-name additions and missing components.
- **float32/int16 downcasting + 200K-row chunked processing** — concrete implementation of the memory-bounded design already committed to for Stages 2–4.
- **Capped hard-negative mining**: top-K non-matching candidates per S1 entity by TF-IDF distance, capped at a 6:1 negative:positive ratio (Stage 4). Genuine improvement over "all blocking survivors as negatives" — with a 3-way blocking union, uncapped negatives could balloon and skew training toward easy negatives.
- **Connected-component cap for exact assignment**: |V| ≤ 100 nodes → `linear_sum_assignment`, else greedy highest-score-first (Stage 5). This directly resolves the connected-component-size risk flagged above.
- **Singleton dual-threshold safeguard**: for candidates scoring in \[τ\_global, τ\_singleton\], require strict physical verification (street number *and* postal code match) before accepting; otherwise collapse to empty. Well-targeted given F0.5's harsh penalty on false singleton merges — added to Stage 5.

**Adopted as tunable starting points, not fixed values** — validated empirically at the sample-dev stage rather than hardcoded:

- TF-IDF `top_n=15`, similarity threshold `0.35`, char n-gram range 3–5 (vs. the original 2–4)
- LightGBM `max_depth=7`, `num_leaves=63`, `learning_rate=0.05`, `subsample=0.8`
- `τ_singleton` (the upper bound of the borderline-confidence band)

These are reasonable priors but weren't derived from our actual data distribution — they get the same treatment as every other hyperparameter in this plan: validated against real recall/F0.5 numbers on the sample split, adjusted if they don't hold up.

**Skipped for now:**

- **Entropy-based positional suffix mining** — the proposed refinement weights legal-suffix candidates by positional entropy across the combined corpus. A simpler frequency+position heuristic (already planned) is easier to debug and likely sufficient; entropy weighting is a candidate follow-up only if suffix-stripping quality proves inadequate on the sample.

## Implementation Results (Dev Sample)

The pipeline described above was fully built and run on a dev sample (20,000 Source 1 train entities, stratified split: 14,000 train / 6,000 held-out validation, real ground truth). Several real findings corrected the plan along the way.

**EDA corrections:**

- **This dataset's `business_address` field almost never contains an actual postal/PIN/ZIP code.** Manual inspection showed the extracted "postal" digit-runs were overwhelmingly house/plot/unit numbers (e.g. `KH NO. -570/13`), not postal codes. Downgraded postal-match to a weak secondary feature rather than a blocking key (see `preprocess.py`'s comment above `_POSTAL_RE`).
- **Locality extraction broke under address-component reordering**, an explicit noise pattern the challenge calls out. Fixed by returning a bag of locality tokens (city/state/region, order-independent) for Jaccard-style comparison instead of picking one segment by position.

**A serious hard-negative-mining bug, found via a concrete example:** the first end-to-end run scored VAL macro F0.5 = 0.607 (precision 35.8%, recall 92.0%) - weak for a precision-weighted metric. Inspecting the worst offender (one Source 1 entity with 55 predicted matches against 3 true ones) showed the false matches shared only city/state/country and a generic "Private Limited" suffix pattern with completely different business names. Root cause: hard negatives were selected by name-token-similarity alone, which systematically excluded exactly this failure mode (locality-confusable, name-different candidates rank as "easy" under a name-only similarity measure) from training. Fixed by ranking hard negatives under BOTH name-token and locality-token similarity and taking the union - the model then sees this exact confusion pattern during training. Re-run: **VAL macro F0.5 = 0.896** (precision 88.5%, recall 87.8%) on the same 6,000-entity split - confirmed at this scale after first validating the fix on a smaller 4,000-entity sample (F0.5 0.926 there).

**A feature-computation performance bug:** the pairwise feature loop profiled at \~2,900 pairs/sec, projecting to \~1 hour per 10M pairs - infeasible for both the dev loop and any full-scale run. Root cause: extracting DataFrame columns via `.values` on pandas 3.x's PyArrow-backed columns has a slow per-element `__getitem__` (profiled at \~14s of a 30s run for 1.1M accesses). Switching to `.tolist()` once per batch fixed it: **41,188 pairs/sec**, a \~14x speedup.

**A blocking candidate-volume problem:** the recall-first blocking ensemble hits 99.66% pair recall, but at a mean of \~6,553 candidates per entity - at full test-set scale (1.73M entities) that's 10+ billion pairs to feature-score, tens of hours even at the fixed throughput. Fixed by capping each blocking strategy's own per-entity contribution (ranked by shared-token count, not a separate re-ranking pass - a first attempt at post-hoc TF-IDF re-ranking was both too slow, \~10ms/row from CSR sub-matrix extraction, and biased against the locality/address-driven candidates those strategies exist to catch). At cap=100: 98.7% recall at \~187 candidates/entity, a \~27x volume cut for a 1.3pp recall cost. Validated on the small sample; not yet re-confirmed at the 20K-entity scale (the large confirmatory run above used the pre-cap blocking).

**Validated output format:** `utils/validate_submission.py` passes with `--check-ids` on both the small- and large-sample runs.

**Not yet done:** `full` mode (real 2.2M-train/1.73M-test run) is still a deliberate `NotImplementedError` - it needs the reference-pool TF-IDF matrix sharded (by country is the natural partition) to fit the \~10M-row S2+S3 pool in 8GB RAM; the per-entity blocking cap above is a prerequisite that's now in place. `Documentation_template.md` is not yet filled in.
