#!/usr/bin/env python3
"""
price_loader.py
Dynamic raw-material price / CO2-factor loader.

ARCHITECTURE (why this file exists)
------------------------------------
The forward (prediction) model learns Composition -> Properties. It has
NOTHING to do with raw-material price. Prices change every month (sometimes
every batch); properties given a fixed composition do not. So price/CO2 must
NOT live inside the training dataset or the model — they must live in an
external, independently-updatable database that the OPTIMIZER reads at
run time.

This module is that external database. It scans a folder of dated CSV
files (one snapshot per update) and always loads the most recent one, e.g.:

    data/cost/17_April_2026_Cost.csv
    data/cost/20_March_2026_Cost.csv
    data/co2/19_April_2026_CO2.csv
    data/co2/01_Jan_2026_CO2.csv

Old files are NOT deleted — they are kept as an audit trail / price history.
Whoever updates the price just drops a new dated CSV into the folder; no
code change, no retraining, nothing else to touch.

CSV SCHEMA
----------
Cost file   : columns -> Material, Price_Tk_per_kg [, Effective_Date]
CO2 file    : columns -> Material, CO2_kg_per_kg    [, Effective_Date]

The optional Effective_Date column (inside the file) is cross-checked
against the date encoded in the filename as a redundant sanity check;
if they disagree, a warning is raised but the filename date always wins
for "which file is latest" purposes (the file the user actually saved
is the one they intend to use "as of now").

USAGE
-----
    from price_loader import load_price_table

    cost_dict, cost_info = load_price_table(
        folder="data/cost", keyword="Cost", value_col="Price_Tk_per_kg",
        materials=materials, fallback=fallback_cost_dict,
    )
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Filename date formats we try to parse, in order ────────────────────────
# Supports things like "17_April_2026_Cost.csv", "2026-04-17_Cost.csv",
# "17-04-2026_Cost.csv", "17_Apr_2026_Cost.csv".
_DATE_TOKEN_RE = re.compile(
    r"(\d{1,2}[_\-][A-Za-z]{3,9}[_\-]\d{4}|\d{4}[_\-]\d{1,2}[_\-]\d{1,2}|\d{1,2}[_\-]\d{1,2}[_\-]\d{4})"
)

_DATE_FORMATS = [
    "%d_%B_%Y", "%d_%b_%Y", "%d-%B-%Y", "%d-%b-%Y",
    "%Y_%m_%d", "%Y-%m-%d",
    "%d_%m_%Y", "%d-%m-%Y",
]


def _parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract a date from a filename such as '17_April_2026_Cost.csv'.
    Returns None (with a warning) if no recognisable date token is found,
    in which case the caller should fall back to file-modified-time.
    """
    match = _DATE_TOKEN_RE.search(filename)
    if not match:
        return None
    token = match.group(1)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None


def _resolve_file_date(path: Path) -> datetime:
    """
    Best-effort date resolution for a single file:
    1) Parse date encoded in the filename (preferred, explicit, auditable).
    2) Fall back to the file's last-modified time on disk, with a warning.
    """
    dt = _parse_date_from_filename(path.name)
    if dt is not None:
        return dt
    warnings.warn(
        f"Could not parse a date from filename '{path.name}'. "
        "Falling back to file-modified-time. Rename the file to include "
        "a date, e.g. '17_April_2026_Cost.csv', for a reliable audit trail."
    )
    return datetime.fromtimestamp(path.stat().st_mtime)


def find_latest_file(folder: Path, keyword: str) -> Optional[Path]:
    """
    Scan `folder` for files whose name contains `keyword` (case-insensitive,
    e.g. 'Cost' or 'CO2') and return the path of the most recent one,
    determined by the date encoded in the filename (fallback: mtime).
    Returns None if the folder doesn't exist or no matching file is found.
    """
    folder = Path(folder)
    if not folder.exists():
        return None

    candidates = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".csv" and keyword.lower() in f.name.lower()
    ]
    if not candidates:
        return None

    dated = [(_resolve_file_date(f), f) for f in candidates]
    dated.sort(key=lambda pair: pair[0])
    return dated[-1][1]


def load_price_table(
    folder,
    keyword: str,
    value_col: str,
    materials: list[str],
    fallback: Optional[dict] = None,
    material_col: str = "Material",
    date_col: str = "Effective_Date",
) -> tuple[dict, dict]:
    """
    Load the latest dated CSV in `folder` matching `keyword` and return
    (values_dict, info_dict).

    values_dict : {material: float value}, one entry per material.
                  Materials missing from the CSV fall back to `fallback`
                  (or 0.0 if no fallback given) with a warning.
    info_dict   : {"file": <path or None>, "as_of": <datetime or None>,
                   "source": "file" | "fallback"}

    This function performs NO retraining and touches NOTHING about the
    prediction model — it is purely a runtime data source for the
    optimizer's objective function.
    """
    fallback = fallback or {}
    latest = find_latest_file(folder, keyword)

    if latest is None:
        warnings.warn(
            f"No dated '{keyword}' CSV found in '{folder}'. "
            "Using fallback values (e.g. from metadata.json)."
        )
        values = {m: float(fallback.get(m, 0.0)) for m in materials}
        return values, {"file": None, "as_of": None, "source": "fallback"}

    df = pd.read_csv(latest)
    if material_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"'{latest.name}' must contain columns '{material_col}' and "
            f"'{value_col}'. Found: {list(df.columns)}"
        )
    table = dict(zip(df[material_col].astype(str), df[value_col].astype(float)))

    # Redundant internal-date vs filename-date cross-check (robustness) — see [note]
    filename_date = _resolve_file_date(latest)
    if date_col in df.columns and not df[date_col].isna().all():
        try:
            internal_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not internal_dates.empty:
                internal_date = internal_dates.max()
                if abs((internal_date - filename_date).days) > 0:
                    warnings.warn(
                        f"'{latest.name}': filename date ({filename_date.date()}) "
                        f"differs from internal '{date_col}' ({internal_date.date()}). "
                        "Using the filename date to determine recency."
                    )
        except Exception:
            pass

    values = {}
    missing = []
    for m in materials:
        if m in table:
            values[m] = float(table[m])
        else:
            missing.append(m)
            values[m] = float(fallback.get(m, 0.0))
    if missing:
        warnings.warn(
            f"'{latest.name}' is missing values for {missing}; "
            "used fallback values for those materials."
        )

    return values, {"file": latest, "as_of": filename_date, "source": "file"}
