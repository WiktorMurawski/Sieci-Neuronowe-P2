"""
Punkt 3: Porównanie large-25x25 vs large-25x50
Punkt 5: Porównanie zbioru z podobnymi elementami vs zbioru zróżnicowanego
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np

from hopfield import HopfieldNetwork
from data import load_dataset, project_root

SEEDS = [0, 1, 2, 3, 4]
NOISE = [0.0, 0.10, 0.20, 0.30, 0.40]


def _recovery_summary(name: str, rule: str = "hebb", n_seeds: int = 5, n_trials: int = 8) -> Dict:
    patterns, shape = load_dataset(name)
    n = patterns.shape[1]
    results = {lvl: [] for lvl in NOISE}

    for seed in SEEDS[:n_seeds]:
        np.random.seed(seed)
        net = HopfieldNetwork(n)
        net.train(patterns, rule=rule, shape=shape, oja_epochs=25)
        for p in patterns:
            for lvl in NOISE:
                for _ in range(n_trials):
                    noisy = net.corrupt(p, lvl)
                    rec, _ = net.update(noisy, mode="async", max_steps=50)
                    results[lvl].append(float(np.dot(rec, p) / n))

    summary = {}
    for lvl, vals in results.items():
        arr = np.array(vals)
        summary[lvl] = {
            "overlap_mean": float(np.mean(arr)),
            "overlap_std": float(np.std(arr)),
            "perfect_rate": float(np.mean(arr > 0.99)),
        }
    return summary


def run_compare_25x25_vs_25x50(
    n_seeds: int = 5,
    out_dir: Path | None = None,
) -> Dict:
    if out_dir is None:
        out_dir = project_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  [compare] large-25x25 vs large-25x50 ...")
    res_25 = _recovery_summary("large-25x25", "hebb", n_seeds=n_seeds)
    res_50 = _recovery_summary("large-25x50", "hebb", n_seeds=n_seeds)

    from experiments.stability import _train
    stab_25 = _train("large-25x25", "hebb").stability_report()
    stab_50 = _train("large-25x50", "hebb").stability_report()

    lines = [
        "PUNKT 3: large-25x25 vs large-25x50 (prawie identyczne wzorce)",
        "=" * 65,
        "",
        "Stabilność (Hebb):",
        f"  25×25 : {stab_25['n_stable_sync']}/{stab_25['n_patterns']} stabilnych",
        f"  25×50 : {stab_50['n_stable_sync']}/{stab_50['n_patterns']} stabilnych",
        "",
        "Odzyskiwanie (overlap, Hebb, mode=async):",
        f"{'noise':>8}  {'25×25':>18}  {'25×50':>18}",
    ]
    for lvl in NOISE:
        a, b = res_25[lvl], res_50[lvl]
        lines.append(
            f"{lvl:>8.2f}  {a['overlap_mean']:.3f}±{a['overlap_std']:.3f} ({a['perfect_rate']*100:4.1f}%)  "
            f"{b['overlap_mean']:.3f}±{b['overlap_std']:.3f} ({b['perfect_rate']*100:4.1f}%)"
        )
    lines += [
        "",
        "Wniosek do raportu:",
        "  Na zbiorze 25×50 żaden wzorzec nie jest stabilny i sieć nie odzyskuje",
        "  oryginałów nawet przy zerowym szumie. Przyczyna leży w strukturze",
        "  korelacji i efektywnym obciążeniu pamięci.",
    ]

    report_path = out_dir / "03_compare_25x25_vs_25x50.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> zapisano {report_path}")
    return {"25x25": res_25, "25x50": res_50, "stab_25": stab_25, "stab_50": stab_50}


def run_compare_similar_vs_diverse(
    n_seeds: int = 5,
    out_dir: Path | None = None,
) -> Dict:
    if out_dir is None:
        out_dir = project_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("letters-abc-8x12", "podobne (litery A/B/C)"),
        ("animals-14x9", "bardziej zróżnicowane (zwierzęta)"),
        ("small-7x7", "kontrola (małe, proste kształty)"),
    ]

    print("  [compare] similar vs diverse ...")
    lines = [
        "PUNKT 5: Zbiór z podobnymi elementami vs zbiór zróżnicowany",
        "=" * 65,
        "",
    ]
    results = {}

    for name, desc in pairs:
        patterns, shape = load_dataset(name)
        n_pat = len(patterns)
        res_h = _recovery_summary(name, "hebb", n_seeds=n_seeds, n_trials=6)
        res_o = _recovery_summary(name, "oja", n_seeds=n_seeds, n_trials=6)
        results[name] = {"hebb": res_h, "oja": res_o, "n_patterns": n_pat, "desc": desc}

        lines.append(f"{name}  - {desc}  ({n_pat} wzorców, {shape[0]}×{shape[1]})")
        lines.append(f"  {'noise':>6}  {'Hebb overlap':>16}  {'Oja overlap':>16}")
        for lvl in NOISE:
            h, o = res_h[lvl], res_o[lvl]
            lines.append(
                f"  {lvl:>6.2f}  {h['overlap_mean']:.3f}±{h['overlap_std']:.3f}     "
                f"{o['overlap_mean']:.3f}±{o['overlap_std']:.3f}"
            )
        lines.append("")

    lines += [
        "Wniosek do raportu:",
        "  Przy podobnej liczności zbioru o silniej skorelowanych / podobnych",
        "  wzorcach (litery) sieć ma większy problem z rozróżnieniem atraktorów.",
        "  Oja zwykle łagodzi ten efekt lepiej niż czysty Hebb.",
    ]

    report_path = out_dir / "05_similar_vs_diverse.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> zapisano {report_path}")
    return results