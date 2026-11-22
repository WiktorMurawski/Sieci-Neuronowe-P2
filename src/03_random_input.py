#!/usr/bin/env python3
"""
03_random_input.py
Dla każdego zbioru i reguły:
  - generuje kilka całkowicie losowych wektorów ±1
  - uruchamia dynamikę sieci
  - sprawdza, czy wynik jest bliski któremuś ze wzorców uczących
    (lub czy to stan spurious / oscylacja)
Wyniki -> results_random.csv + .txt
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

OUT_CSV = ROOT / "results_random.csv"
OUT_TXT = ROOT / "results_random.txt"

N_RANDOM = 10          # ile losowych wejść na zbiór/regułę
BASE_SEED = 999
UPDATE_MODE = "async"
MAX_STEPS = 100
MATCH_THRESHOLD = 0.05  # jeśli Hamming < 5% n → uznajemy za odtworzenie wzorca


def hamming_frac(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a.flatten() != b.flatten()))


def closest_pattern(state: np.ndarray, patterns: np.ndarray) -> tuple[int, float]:
    """Zwraca (index, hamming_frac) najbliższego wzorca."""
    dists = [hamming_frac(state, p) for p in patterns]
    i = int(np.argmin(dists))
    return i, dists[i]


def main():
    rows = []
    lines = []
    lines.append("=== Reakcja na całkowicie losowe wejście ===\n")
    lines.append(f"n_random={N_RANDOM}, mode={UPDATE_MODE}, max_steps={MAX_STEPS}\n")

    for name in DATASETS:
        patterns, shape = load_dataset(name, base_dir=ATTACH)
        n_pat, n = patterns.shape
        lines.append(f"\n--- {name}  n={n}  #pat={n_pat} ---")

        for rule in ("hebb", "oja"):
            net = HopfieldNetwork(n)
            net.train(patterns, rule=rule, shape=shape)

            recovered_to_stored = 0
            spurious = 0
            energies = []
            closest_list = []

            lines.append(f"  [{rule}]")

            for r in range(N_RANDOM):
                np.random.seed(BASE_SEED + r * 13 + hash(name) % 1000)
                rnd = np.random.choice([-1.0, 1.0], size=n)
                final, hist = net.update(rnd, mode=UPDATE_MODE, max_steps=MAX_STEPS, track=True)
                e = net.energy(final)
                energies.append(e)

                idx, dist = closest_pattern(final, patterns)
                closest_list.append((idx, dist))

                if dist <= MATCH_THRESHOLD:
                    recovered_to_stored += 1
                    tag = f"-> pattern[{idx}] (d={dist:.3f})"
                else:
                    spurious += 1
                    tag = f"-> spurious (closest pat[{idx}] d={dist:.3f})"

                # proste wykrycie oscylacji (sync-like: ostatnie dwa stany różne i powtarzające się)
                osc = False
                if len(hist) >= 3:
                    if np.array_equal(hist[-1], hist[-3]) and not np.array_equal(hist[-1], hist[-2]):
                        osc = True
                        tag += " [OSC?]"

                lines.append(f"    rnd[{r}]: E={e:8.1f}  steps≈{len(hist)-1}  {tag}")

            mean_e = float(np.mean(energies))
            frac_stored = recovered_to_stored / N_RANDOM

            row = {
                "dataset": name,
                "rule": rule,
                "n_neurons": n,
                "n_patterns": n_pat,
                "n_random": N_RANDOM,
                "frac_to_stored": round(frac_stored, 3),
                "n_spurious": spurious,
                "mean_energy_final": round(mean_e, 2),
            }
            rows.append(row)

            lines.append(
                f"    >>> frac_to_stored={frac_stored:.2f}  "
                f"spurious={spurious}/{N_RANDOM}  mean_E={mean_e:.1f}"
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
