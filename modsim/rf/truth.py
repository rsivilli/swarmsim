"""Ground-truth table writers (Parquet when pyarrow is installed, CSV otherwise)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def have_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_format(truth_format: str) -> str:
    """'auto' picks parquet when pyarrow is available; an explicit 'parquet' requires it."""
    if truth_format == "auto":
        return "parquet" if have_pyarrow() else "csv"
    if truth_format == "parquet" and not have_pyarrow():
        raise RuntimeError(
            "Parquet truth tables need pyarrow: install the rf extra (uv sync --extra rf) "
            "or use --truth-format csv"
        )
    return truth_format


def write_table(path_stem: Path, columns: dict[str, np.ndarray | list], fmt: str) -> Path:
    """Write equal-length columns to ``<path_stem>.parquet`` or ``<path_stem>.csv``."""
    lengths = {len(v) for v in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"Columns of {path_stem.name} have different lengths: {lengths}")
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = path_stem.with_suffix(".parquet")
        pq.write_table(pa.table({k: _to_arrow(v) for k, v in columns.items()}), path)
        return path

    path = path_stem.with_suffix(".csv")
    names = list(columns)
    cols = [np.asarray(columns[k]).tolist() for k in names]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(zip(*cols))
    return path


def _to_arrow(v):
    return v if isinstance(v, list) else np.asarray(v)
