"""
Punkt 4: Skuteczność odzyskiwania oryginałów z uszkodzonego wejścia
w zależności od liczności i rodzaju zbioru.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np

from hopfield import HopfieldNetwork
from data import load_dataset, DATASETS, project_root

SEEDS = [0, 1, 2, 3, 4]
DEFAULT_NOISE = [0.0, 0.10, 0.20, 0.30, 0.40]


def run_recovery(
    datasets: List[str] | None = None,
    rules: List[str] = ("hebb", "oja"),
    noise_levels: List[float] | None = None,
    n_seeds: int = 5,
    n_trials_per_pattern: int = 6,
    mode: str = "async",
    out_dir: Path | None = None,
) -> Dict:
    if datasets is None:
        datasets = [d for d in DATASETS if DATASETS[d]["shape"][0] * DATASETS[d]["shape"][1] <= 700]
    if noise_levels is None:
        noise_levels = DEFAULT_NOISE
    if out_dir is None:
        out_dir = project_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    lines = [
        "PUNKT 4: Odzyskiwanie z zaszumionego wejścia",
        "=" * 65,
        f"Ziarna: {SEEDS[:n_seeds]}, prób na wzorzec: {n_trials_per_pattern}, mode={mode}",
        "",
    ]

    for name in datasets:
        patterns, shape = load_dataset(name)
        n_pat, n = patterns.shape
        print(f"  [recovery] {name} ...")

        all_results[name] = {}
        lines.append(f"--- {name}  ({n_pat} wzorców, N={n}) ---")

        for rule in rules:
            overlaps = {lvl: [] for lvl in noise_levels}
            perfects = {lvl: [] for lvl in noise_levels}

            for seed in SEEDS[:n_seeds]:
                np.random.seed(seed)
                net = HopfieldNetwork(n)
                net.train(patterns, rule=rule, shape=shape, oja_eta=0.001, oja_epochs=35)

                for p in patterns:
                    for lvl in noise_levels:
                        for _ in range(n_trials_per_pattern):
                            noisy = net.corrupt(p, lvl)
                            recovered, _ = net.update(noisy, mode=mode, max_steps=60)
                            ov = float(np.dot(recovered, p) / n)
                            overlaps[lvl].append(ov)
                            perfects[lvl].append(1.0 if ov > 0.99 else 0.0)

            summary = {}
            lines.append(f"  reguła = {rule}")
            for lvl in noise_levels:
                ov = np.array(overlaps[lvl])
                pr = np.array(perfects[lvl])
                summary[lvl] = {
                    "overlap_mean": float(np.mean(ov)),
                    "overlap_std": float(np.std(ov)),
                    "perfect_rate": float(np.mean(pr)),
                    "n_samples": len(ov),
                }
                lines.append(
                    f"    noise={lvl:.2f}: overlap = {summary[lvl]['overlap_mean']:.3f} ± "
                    f"{summary[lvl]['overlap_std']:.3f}   "
                    f"perfect = {summary[lvl]['perfect_rate']*100:.1f}%"
                )
            all_results[name][rule] = summary
            lines.append("")

    report_path = out_dir / "04_recovery.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> zapisano {report_path}")
    return all_results