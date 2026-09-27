"""
I/O helpers shared across pipeline stages.

Design goals (see Documentation_template.md / the plan doc for rationale):
  - Never load a full Source-2/3 file as a dense pandas object-dtype frame at
    full scale if it can be avoided — stream in chunks instead.
  - Read every TSV as plain strings (na_filter=False) so missing values are
    the empty string, never NaN — this avoids float-casting bugs on an
    entity_id-like column and keeps downstream code simple.
  - Cache normalized frames to Parquet so a re-run of a later stage doesn't
    have to re-parse and re-normalize a 500MB TSV.
"""

from __future__ import annotations

import gc
import os
from typing import Iterator, Optional, Sequence

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

DEFAULT_CHUNKSIZE = 200_000


def read_source_tsv(path: str, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a full source TSV (S1/S2/S3) as string dtype, empty string for missing."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        na_filter=False,
        usecols=usecols,
        engine="c",
    )
    return df


def iter_source_tsv_chunks(
    path: str,
    usecols: Optional[Sequence[str]] = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Iterator[pd.DataFrame]:
    """Stream a source TSV in chunks (for the large S2/S3 files at full scale)."""
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        na_filter=False,
        usecols=usecols,
        engine="c",
        chunksize=chunksize,
    )
    for chunk in reader:
        yield chunk


def read_ground_truth(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    return df


def write_id_list_tsv(df: pd.DataFrame, path: str, id_col: str, list_col: str) -> None:
    """Write a matching_results.tsv / candidate_pairs.tsv - shaped file.

    ``df`` must have ``id_col`` (str) and ``list_col`` (an already-comma-joined
    string, empty string for no matches/candidates).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = df[[id_col, list_col]]
    out.to_csv(path, sep="\t", index=False, encoding="utf-8")


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def maybe_load_cached(path: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(path):
        return load_parquet(path)
    return None


def collect() -> None:
    """Thin wrapper so call sites read as an explicit memory-management step."""
    gc.collect()
