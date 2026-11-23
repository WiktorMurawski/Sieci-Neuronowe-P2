"""
Ładowanie zbiorów wzorców i pomocnicze ścieżki projektu.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


DATASETS = {
    "small-7x7":        {"file": "small-7x7.csv",        "shape": (7, 7)},
    "letters-abc-8x12": {"file": "letters-abc-8x12.csv", "shape": (12, 8)},
    "animals-14x9":     {"file": "animals-14x9.csv",     "shape": (9, 14)},
    "letters-14x20":    {"file": "letters-14x20.csv",    "shape": (20, 14)},
    "OCRA-12x30":       {"file": "OCRA-12x30-cut.csv",   "shape": (30, 12)},
    "large-25x25":      {"file": "large-25x25.csv",      "shape": (25, 25)},
    "large-25x25.plus": {"file": "large-25x25.plus.csv", "shape": (25, 25)},
    "large-25x50":      {"file": "large-25x50.csv",      "shape": (50, 25)},
    # własne zbiory 
    "stable-5x5":       {"file": "stable-5x5.csv",       "shape": (5, 5)},
    # "cats-200x300":     {"file": "cats-200x300.csv",     "shape": (300, 200)},
}


def load_dataset(name: str, base_dir: str | Path | None = None) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Zwraca (patterns, shape).
    patterns ma kształt (n_patterns, n_neurons), wartości +1 lub -1.
    """
    if name not in DATASETS:
        raise KeyError(f"Nieznany zbiór: {name}. Dostępne: {list(DATASETS.keys())}")
    if base_dir is None:
        base_dir = project_root() / "patterns"
    info = DATASETS[name]
    path = Path(base_dir) / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku: {path}")
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    shape = info["shape"]
    expected = shape[0] * shape[1]
    if data.shape[1] != expected:
        raise ValueError(f"Niezgodność wymiarów dla {name}: {data.shape[1]} != {expected}")
    data = np.where(data > 0, 1.0, -1.0)
    return data, shape


def list_datasets() -> None:
    print("Dostępne zbiory:")
    for k, v in DATASETS.items():
        print(f"  {k:20s} -> {v['file']:25s}  {v['shape']}")