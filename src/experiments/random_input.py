"""
Punkt 6: Jak nauczona sieć reaguje na całkowicie losowe wejście?
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np

from hopfield import HopfieldNetwork
from data import load_dataset, DATASETS, project_root

SEEDS = [0, 1, 2, 3, 4]


def run_random_input(
    datasets: List[str] | None = None,
    rule: str = "hebb",
    n_seeds: int = 5,
    n_random_per_seed: int = 12,
    out_dir: Path | None = None,
) -> Dict:
    if datasets is None:
        datasets = [d for d in DATASETS if DATASETS[d]["shape"][0] * DATASETS[d]["shape"][1] <= 700]
    if out_dir is None:
        out_dir = project_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    lines = [
        "PUNKT 6: Reakcja na całkowicie losowe wejście",
        "=" * 65,
        f"Reguła: {rule}, ziarna: {SEEDS[:n_seeds]}, prób na seed: {n_random_per_seed}",
        "",
    ]

    for name in datasets:
        patterns, shape = load_dataset(name)
        n = patterns.shape[1]
        print(f"  [random] {name} ...")

        attractors = []
        hits = 0
        total = 0

        for seed in SEEDS[:n_seeds]:
            np.random.seed(seed)
            net = HopfieldNetwork(n)
            net.train(patterns, rule=rule, shape=shape)

            for _ in range(n_random_per_seed):
                rand = np.random.choice([-1.0, 1.0], size=n)
                final, _ = net.update(rand, mode="async", max_steps=80)
                for p in patterns:
                    if np.array_equal(final, p) or np.array_equal(final, -p):
                        hits += 1
                        break
                attractors.append(tuple(final.tolist()))
                total += 1

        unique = len(set(attractors))
        res = {
            "n_trials": total,
            "unique_attractors": unique,
            "hit_training_rate": hits / total if total else 0.0,
        }
        all_results[name] = res

        lines.append(f"{name}  (N={n}, {len(patterns)} wzorców)")
        lines.append(f"  prób: {total}")
        lines.append(f"  unikalnych atraktorów: {unique}")
        lines.append(f"  trafienie w wzorzec uczący (lub negację): {res['hit_training_rate']*100:.1f}%")
        lines.append("")

    report_path = out_dir / "06_random_input.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → zapisano {report_path}")
    return all_results