"""Lightweight CSV adaptation shared by validation, training and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _clean_labels(series: pd.Series) -> pd.Series:
    # Preserve missing values so all consumers can report incomplete labels.
    return series.map(lambda value: str(value).strip() if pd.notna(value) else value)


def read_dataset_csv(path: str | Path, heads: Iterable[str]) -> pd.DataFrame:
    """Read either canonical columns or image_path CSVs without rewriting them.

    Paths remain relative when provided that way: DATA_ROOT is applied by the
    dataset consumer. Extra metadata columns are retained.
    """
    frame = pd.read_csv(path)
    if "image_path" in frame.columns:
        roots: list[object] = []
        names: list[object] = []
        for value in frame["image_path"]:
            if pd.isna(value) or not str(value).strip():
                roots.append(value)
                names.append(value)
            else:
                image = Path(str(value).strip())
                roots.append(str(image.parent))
                names.append(image.name)
        for column, values in (("root_folder", roots), ("image_name", names)):
            derived = pd.Series(values, index=frame.index)
            if column in frame.columns:
                # Conflicting path representations can otherwise evaluate the
                # wrong files while silently accepting the CSV.
                existing = _clean_labels(frame[column])
                matching = (existing == derived) | (existing.isna() & derived.isna())
                if not matching.all():
                    raise ValueError(f"{path}: {column} conflicts with image_path.")
            else:
                frame[column] = derived
    requested_heads = list(heads)
    if "vehicle_type" in requested_heads and "type" in frame.columns:
        aliases = _clean_labels(frame["type"])
        if "vehicle_type" in frame.columns:
            canonical = _clean_labels(frame["vehicle_type"])
            matching = (canonical == aliases) | (canonical.isna() & aliases.isna())
            if not matching.all():
                raise ValueError(f"{path}: vehicle_type conflicts with type.")
        else:
            frame["vehicle_type"] = aliases
    for head in requested_heads:
        if head in frame.columns:
            frame[head] = _clean_labels(frame[head])
    return frame
