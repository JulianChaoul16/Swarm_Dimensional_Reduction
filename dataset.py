"""Load a header-based CSV containing numeric features and class labels."""

import csv
from pathlib import Path

import numpy as np


def load_csv(path, target, delimiter=","):
    """Return X, y, feature names; exclude the named target from the features."""
    if len(delimiter) != 1:
        raise ValueError("The delimiter must be one character.")
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            raise ValueError("The CSV is empty or has no header.")
        header = [name.strip() for name in header]
        if any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError("CSV column names must be nonempty and unique.")
        if target not in header:
            raise ValueError(f"Target column {target!r} not found. Available columns: {', '.join(header)}")
        feature_names = [name for name in header if name != target]
        if not feature_names:
            raise ValueError("The CSV needs at least one feature column besides the target.")
        target_index = header.index(target)
        rows, labels = [], []
        for row in reader:
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV line {reader.line_num}: expected {len(header)} values, got {len(row)}.")
            label = row[target_index].strip()
            if not label:
                raise ValueError(f"CSV line {reader.line_num}: target value is missing.")
            values = []
            for index, raw in enumerate(row):
                if index == target_index:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    raise ValueError(f"CSV line {reader.line_num}, column {header[index]!r}: expected a numeric value.") from None
                if not np.isfinite(value):
                    raise ValueError(f"CSV line {reader.line_num}, column {header[index]!r}: values must be finite.")
                values.append(value)
            rows.append(values)
            labels.append(label)
    if not rows:
        raise ValueError("The CSV contains no data rows.")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Classification requires at least two target classes.")
    if counts.min() < 2:
        raise ValueError("Each target class needs at least two rows for a stratified split.")
    return np.asarray(rows, dtype=float), np.asarray(labels), feature_names
