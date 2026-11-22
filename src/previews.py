"""Zapis wizualizacji wzorców (pomocnicze, używane przez wszystkie eksperymenty)."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data import project_root, DATASETS, load_dataset


def save_previews(name: str, patterns: np.ndarray, shape: tuple[int, int]) -> Path:
    """
    Zapisuje każdy wzorzec jako PNG + siatkę zbiorczą.
    Zwraca ścieżkę do katalogu z podglądami.
    """
    dest = project_root() / "previews" / name
    dest.mkdir(parents=True, exist_ok=True)

    n = len(patterns)
    for i, p in enumerate(patterns):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(p.reshape(shape), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"{name} #{i}")
        ax.axis("off")
        fig.savefig(dest / f"pattern_{i:02d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    for i, p in enumerate(patterns):
        axes[i].imshow(p.reshape(shape), cmap="gray", vmin=-1, vmax=1)
        axes[i].set_title(f"#{i}", fontsize=9)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"{name}  ({n} wzorców, {shape[0]}×{shape[1]})", fontsize=12)
    fig.tight_layout()
    fig.savefig(dest / "_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dest

if __name__ == "__main__":
    print("\n=== Generowanie podglądów wzorców ===")
    for name in DATASETS:
        patterns, shape = load_dataset(name)
        dest = save_previews(name, patterns, shape)
        print(f"  {name:20s} -> {dest}")