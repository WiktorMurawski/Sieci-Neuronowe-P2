#!/usr/bin/env python3
"""
02_recovery.py
Skuteczność odzyskiwania oryginału z uszkodzonego wejścia.
Dla każdego zbioru i reguły:
  - poziomy szumu (frakcja flipowanych bitów)
  - kilka seedów
  - średnia + odchylenie standardowe sukcesu (dokładne odtworzenie)
Wyniki -> results_recovery.csv + .txt
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import csv

ROOT = Path(__file__).resolve().parent
ATTACH = Path("/home/workdir/attachments")
sys.path.insert(0, str(ATTACH))

from data import DATASETS, load_dataset
from hopfield import HopfieldNetwork

OUT_CSV = ROOT / "results_recovery.csv"
OUT_TXT = ROOT / "results_recovery.txt"

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
N_SEEDS = 5
BASE_SEED = 123
UPDATE_MODE = "async"   # "sync" lub "async"
MAX_STEPS = 80


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a.flatten() != b.flatten()))


def main():
    rows = []
    lines = []
    lines.append("=== Skuteczność odzyskiwania z uszkodzonego wejścia ===\n")
    lines.append(f"mode={UPDATE_MODE}, max_steps={MAX_STEPS}, n_seeds={N_SEEDS}\n")

    for name in DATASETS:
        patterns, shape = load_dataset(name, base_dir=ATTACH)
        n_pat, n = patterns.shape
        lines.append(f"\n--- {name}  n={n}  #pat={n_pat} ---")

        for rule in ("hebb", "oja"):
            net = HopfieldNetwork(n)
            net.train(patterns, rule=rule, shape=shape)

            lines.append(f"  [{rule}]")

            for noise in NOISE_LEVELS:
                successes = []   # 0/1 per (pattern, seed)
                final_dists = []

                for pi, p in enumerate(patterns):
                    for s in range(N_SEEDS):
                        np.random.seed(BASE_SEED + pi * 100 + s * 7 + int(noise * 1000))
                        corrupted = net.corrupt(p, noise)
                        recovered, _ = net.update(corrupted, mode=UPDATE_MODE, max_steps=MAX_STEPS)
                        ok = int(np.array_equal(recovered, p))
                        successes.append(ok)
                        final_dists.append(hamming(recovered, p) / n)

                mean_succ = float(np.mean(successes))
                std_succ = float(np.std(successes))
                mean_dist = float(np.mean(final_dists))

                row = {
                    "dataset": name,
                    "rule": rule,
                    "noise": noise,
                    "n_neurons": n,
                    "n_patterns": n_pat,
                    "mean_success": round(mean_succ, 4),
                    "std_success": round(std_succ, 4),
                    "mean_final_hamming_frac": round(mean_dist, 4),
                    "n_trials": len(successes),
                }
                rows.append(row)

                lines.append(
                    f"    noise={noise:.2f}: success={mean_succ*100:5.1f}% ± {std_succ*100:4.1f}%  "
                    f"(mean final Hamming frac={mean_dist:.3f})"
                )

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
