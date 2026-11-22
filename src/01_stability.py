#!/usr/bin/env python3
"""
01_stability.py
Dla każdego zbioru wzorców trenuje sieć regułą Hebba i Oji,
sprawdza ile wzorców uczących jest stabilnych (sync + async)
i zapisuje wyniki do results_stability.csv oraz .txt.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import csv

# ścieżki
ROOT = Path(__file__).resolve().parent.parent
ATTACH = Path("patterns")
sys.path.insert(0, str(ATTACH))

from data import DATASETS, load_dataset
from hopfield import HopfieldNetwork

OUT_DIR = ROOT / "results"

OUT_CSV = OUT_DIR / "results_stability.csv"
OUT_TXT = OUT_DIR / "results_stability.txt"

# ile razy sprawdzać stabilność async (losowa kolejność neuronów)
N_ASYNC_TRIALS = 5
SEED = 42


def check_async_stable(net: HopfieldNetwork, pattern: np.ndarray, trials: int = N_ASYNC_TRIALS) -> float:
    """Zwraca ułamek prób, w których wzorzec pozostał niezmieniony (async)."""
    ok = 0
    for t in range(trials):
        np.random.seed(SEED + t * 17)
        if net.is_stable(pattern, mode="async"):
            ok += 1
    return ok / trials


def main():
    rows = []
    lines = []
    lines.append("=== Stabilnosc wzorcow uczacych (Hebb vs Oja) ===")

    for name in DATASETS:
        patterns, shape = load_dataset(name, base_dir=ATTACH)
        n_pat, n = patterns.shape
        lines.append(f"\n--- {name}  |  n={n}  |  #wzorców={n_pat}  |  shape={shape} ---")

        for rule in ("hebb", "oja"):
            net = HopfieldNetwork(n)
            net.train(patterns, rule=rule, shape=shape)

            n_stable_sync = 0
            async_scores = []
            energies = []

            for i, p in enumerate(patterns):
                stable_s = net.is_stable(p, mode="sync")
                if stable_s:
                    n_stable_sync += 1
                frac_a = check_async_stable(net, p)
                async_scores.append(frac_a)
                energies.append(net.energy(p))

            mean_async = float(np.mean(async_scores))
            n_stable_async_approx = sum(1 for s in async_scores if s >= 0.8)

            row = {
                "dataset": name,
                "rule": rule,
                "n_neurons": n,
                "n_patterns": n_pat,
                "n_stable_sync": n_stable_sync,
                "frac_stable_sync": n_stable_sync / n_pat,
                "mean_async_stability": round(mean_async, 3),
                "n_stable_async_approx": n_stable_async_approx,
                "mean_energy": round(float(np.mean(energies)), 2),
            }
            rows.append(row)

            lines.append(
                f"  {rule:5s}: stable_sync={n_stable_sync}/{n_pat} "
                f"({100*n_stable_sync/n_pat:.0f}%)  |  "
                f"mean_async_stab={mean_async:.2f}  |  "
                f"approx_stable_async={n_stable_async_approx}  |  "
                f"mean_E={row['mean_energy']}"
            )
            # szczegóły per wzorzec
            for i, (e, a) in enumerate(zip(energies, async_scores)):
                lines.append(f"      pat[{i}]: E={e:.1f}  async_stab={a:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # zapis CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nZapisano: {OUT_CSV}")
    print(f"Zapisano: {OUT_TXT}")


if __name__ == "__main__":
    main()
