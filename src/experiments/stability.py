"""
Punkt 1 i 2: Skuteczność reguł uczenia (Hebb / Oja)
oraz stabilność wzorców uczących na poszczególnych zbiorach.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np

from hopfield import HopfieldNetwork
from data import load_dataset, DATASETS, project_root

SEEDS = [0, 1, 2, 3, 4]


def _train(name: str, rule: str, seed: int | None = None) -> HopfieldNetwork:
    patterns, shape = load_dataset(name)
    if seed is not None:
        np.random.seed(seed)
    net = HopfieldNetwork(patterns.shape[1])
    net.train(patterns, rule=rule, shape=shape, oja_eta=0.001, oja_epochs=40)
    return net


def run_stability(
    datasets: List[str] | None = None,
    n_seeds: int = 5,
    out_dir: Path | None = None,
) -> Dict:
    if datasets is None:
        datasets = list(DATASETS.keys())
    if out_dir is None:
        out_dir = project_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    lines = [
        "PUNKT 1 + 2: Stabilność wzorców uczących (Hebb vs Oja)",
        "=" * 65,
        f"Ziarna (Oja): {SEEDS[:n_seeds]}",
        "",
    ]

    for name in datasets:
        patterns, shape = load_dataset(name)
        n_pat, n_neu = patterns.shape
        print(f"  [stability] {name} ({n_pat} wzorców, {n_neu} neuronów)...")

        net_h = _train(name, "hebb")
        rep_h = net_h.stability_report()

        oja_counts = []
        for seed in SEEDS[:n_seeds]:
            net_o = _train(name, "oja", seed=seed)
            rep_o = net_o.stability_report()
            oja_counts.append(rep_o["n_stable_sync"])

        res = {
            "n_patterns": n_pat,
            "n_neurons": n_neu,
            "shape": shape,
            "hebb_stable": rep_h["n_stable_sync"],
            "hebb_fraction": rep_h["fraction_stable"],
            "oja_mean": float(np.mean(oja_counts)),
            "oja_std": float(np.std(oja_counts)),
            "oja_raw": oja_counts,
            "oja_fraction": float(np.mean(oja_counts)) / n_pat,
        }
        all_results[name] = res

        lines.append(f"{name}  ({n_pat} wzorców, {shape[0]}×{shape[1]}, N={n_neu})")
        lines.append(f"  Hebb : {res['hebb_stable']:2d}/{n_pat}  ({res['hebb_fraction']*100:5.1f}%)")
        lines.append(f"  Oja  : {res['oja_mean']:.2f} ± {res['oja_std']:.2f} / {n_pat}  "
                     f"({res['oja_fraction']*100:5.1f}%)   raw={oja_counts}")
        lines.append("")

    report_path = out_dir / "01_02_stability.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> zapisano {report_path}")
    return all_results