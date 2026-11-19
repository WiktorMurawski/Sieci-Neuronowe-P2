#!/usr/bin/env python3
"""
Eksperymenty na sieci Hopfielda – pełny pipeline.

Użycie:
    python run_experiments.py                     # wszystkie zbiory
    python run_experiments.py small-7x7 animals-14x9
    python run_experiments.py --list

Zapisuje:
  - podgląd wzorców:  artifacts/previews/<dataset>/pattern_XX.png
  - wyniki tekstowe:  artifacts/results_<dataset>.txt
  - zbiorczy raport:  artifacts/summary.txt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# lokalny import
sys.path.insert(0, str(Path(__file__).parent))
from hopfield import (
    HopfieldNetwork,
    load_dataset,
    train_on_dataset,
    DATASETS,
)

SEEDS = [0, 1, 2, 3, 4]          # 5 ziaren
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
OUT_DIR = Path("/home/workdir/artifacts")
PREVIEW_DIR = OUT_DIR / "previews"
RESULTS_DIR = OUT_DIR / "results"


# ----------------------------------------------------------------------
# Wizualizacja wzorców
# ----------------------------------------------------------------------
def save_pattern_previews(name: str, patterns: np.ndarray, shape: Tuple[int, int]):
    """Zapisuje każdy wzorzec jako osobny PNG + siatkę zbiorczą."""
    dest = PREVIEW_DIR / name
    dest.mkdir(parents=True, exist_ok=True)

    n = len(patterns)
    # pojedyncze
    for i, p in enumerate(patterns):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(p.reshape(shape), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"{name} #{i}")
        ax.axis("off")
        fig.savefig(dest / f"pattern_{i:02d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # siatka
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
    print(f"  [preview] zapisano {n} wzorców + siatkę → {dest}")


# ----------------------------------------------------------------------
# Eksperymenty
# ----------------------------------------------------------------------
def experiment_stability(name: str, n_seeds: int = 5) -> Dict:
    """
    Stabilność wzorców uczących (Hebb + Oja).
    Dla Hebb wynik jest deterministyczny; dla Oja uśredniamy po seedach
    (losowa kolejność prezentacji + inicjalizacja).
    """
    patterns, shape = load_dataset(name)
    patterns = np.where(patterns > 0, 1.0, -1.0)
    n_pat = len(patterns)

    results = {"hebb": [], "oja": []}

    # Hebb – jeden przebieg wystarczy
    net = HopfieldNetwork(patterns.shape[1])
    net.train(patterns, rule="hebb", shape=shape)
    rep = net.stability_report()
    results["hebb"] = {
        "n_stable": rep["n_stable_sync"],
        "fraction": rep["fraction_stable"],
        "energies": [d["energy"] for d in rep["details"]],
    }

    # Oja – kilka seedów
    oja_stable = []
    for seed in SEEDS[:n_seeds]:
        np.random.seed(seed)
        net = HopfieldNetwork(patterns.shape[1])
        net.train(patterns, rule="oja", shape=shape, oja_eta=0.001, oja_epochs=40)
        rep = net.stability_report()
        oja_stable.append(rep["n_stable_sync"])

    results["oja"] = {
        "n_stable_mean": float(np.mean(oja_stable)),
        "n_stable_std": float(np.std(oja_stable)),
        "fraction_mean": float(np.mean(oja_stable)) / n_pat,
        "raw": oja_stable,
    }
    return results


def experiment_recovery(
    name: str,
    rule: str = "hebb",
    noise_levels: List[float] = None,
    n_seeds: int = 5,
    n_trials_per_pattern: int = 8,
    mode: str = "async",
) -> Dict:
    """
    Skuteczność odzyskiwania z zaszumionego wejścia.
    Zwraca dla każdego poziomu szumu: mean_overlap ± std oraz perfect_rate.
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    patterns, shape = load_dataset(name)
    patterns = np.where(patterns > 0, 1.0, -1.0)
    n_pat, n = patterns.shape

    # zbieramy overlap-y ze wszystkich seedów i prób
    all_overlaps: Dict[float, List[float]] = {lvl: [] for lvl in noise_levels}
    all_perfect: Dict[float, List[float]] = {lvl: [] for lvl in noise_levels}

    for seed in SEEDS[:n_seeds]:
        np.random.seed(seed)
        net = HopfieldNetwork(n)
        net.train(patterns, rule=rule, shape=shape,
                  oja_eta=0.001, oja_epochs=35)

        for p in patterns:
            for lvl in noise_levels:
                for _ in range(n_trials_per_pattern):
                    noisy = net.corrupt(p, lvl)
                    recovered, _ = net.update(noisy, mode=mode, max_steps=60)
                    overlap = float(np.dot(recovered, p) / n)
                    all_overlaps[lvl].append(overlap)
                    all_perfect[lvl].append(1.0 if overlap > 0.99 else 0.0)

    summary = {}
    for lvl in noise_levels:
        ov = np.array(all_overlaps[lvl])
        pr = np.array(all_perfect[lvl])
        summary[lvl] = {
            "overlap_mean": float(np.mean(ov)),
            "overlap_std": float(np.std(ov)),
            "perfect_rate": float(np.mean(pr)),
            "perfect_std": float(np.std(pr)),  # przybliżone
            "n_samples": len(ov),
        }
    return summary


def experiment_random_input(name: str, rule: str = "hebb", n_seeds: int = 5, n_random: int = 30) -> Dict:
    """
    Co się dzieje, gdy podamy całkowicie losowe wejście?
    Zliczamy, do ilu różnych atraktorów sieć zbiega i jak często
    trafia w jeden z wzorców uczących.
    """
    patterns, shape = load_dataset(name)
    patterns = np.where(patterns > 0, 1.0, -1.0)
    n = patterns.shape[1]

    attractors = []
    hits_training = 0
    total = 0

    for seed in SEEDS[:n_seeds]:
        np.random.seed(seed)
        net = HopfieldNetwork(n)
        net.train(patterns, rule=rule, shape=shape)

        for _ in range(n_random // n_seeds + 1):
            rand = np.random.choice([-1.0, 1.0], size=n)
            final, _ = net.update(rand, mode="async", max_steps=80)
            # czy to jeden z wzorców uczących?
            for p in patterns:
                if np.array_equal(final, p) or np.array_equal(final, -p):
                    hits_training += 1
                    break
            attractors.append(tuple(final))
            total += 1

    unique = len(set(attractors))
    return {
        "n_trials": total,
        "unique_attractors": unique,
        "hit_training_rate": hits_training / total,
    }


# ----------------------------------------------------------------------
# Raportowanie
# ----------------------------------------------------------------------
def fmt_mean_std(mean: float, std: float, fmt: str = ".3f") -> str:
    return f"{mean:{fmt}} ± {std:{fmt}}"


def run_one_dataset(name: str, do_recovery: bool = True, do_random: bool = True):
    print("\n" + "=" * 70)
    print(f"ZBIÓR: {name}")
    print("=" * 70)

    patterns, shape = load_dataset(name)
    patterns = np.where(patterns > 0, 1.0, -1.0)
    n_pat, n_neu = patterns.shape
    print(f"  wzorców: {n_pat}, neurony: {n_neu}, kształt: {shape}")

    # 1. Wizualizacja
    save_pattern_previews(name, patterns, shape)

    # 2. Stabilność
    print("\n--- Stabilność wzorców uczących ---")
    stab = experiment_stability(name)
    print(f"  Hebb : {stab['hebb']['n_stable']}/{n_pat} stabilnych "
          f"({stab['hebb']['fraction']*100:.1f}%)")
    print(f"  Oja  : {fmt_mean_std(stab['oja']['n_stable_mean'], stab['oja']['n_stable_std'], '.2f')}"
          f" / {n_pat}  ({stab['oja']['fraction_mean']*100:.1f}%)")

    recovery_hebb = recovery_oja = None
    random_info = None

    # 3. Odzyskiwanie (tylko jeśli zbiór nie jest gigantyczny)
    if do_recovery and n_neu <= 700:          # large-25x50 jest wolny
        print("\n--- Odzyskiwanie z szumu (overlap) ---")
        for rule in ["hebb", "oja"]:
            print(f"  reguła = {rule}")
            rec = experiment_recovery(name, rule=rule, n_seeds=5, n_trials_per_pattern=6)
            if rule == "hebb":
                recovery_hebb = rec
            else:
                recovery_oja = rec
            for lvl in [0.0, 0.10, 0.20, 0.30, 0.40]:
                s = rec[lvl]
                print(f"    noise={lvl:.2f}: overlap = {fmt_mean_std(s['overlap_mean'], s['overlap_std'])}"
                      f"   perfect = {s['perfect_rate']*100:.1f}%")
    else:
        print("\n--- Odzyskiwanie: pominięte (zbiór zbyt duży / wyłączone) ---")

    # 4. Losowe wejście
    if do_random and n_neu <= 700:
        print("\n--- Reakcja na losowe wejście ---")
        random_info = experiment_random_input(name, rule="hebb", n_seeds=5, n_random=40)
        print(f"  unikalnych atraktorów: {random_info['unique_attractors']} "
              f"(z {random_info['n_trials']} prób)")
        print(f"  trafienie w wzorzec uczący (lub jego negację): "
              f"{random_info['hit_training_rate']*100:.1f}%")

    # zapis wyników
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Zbiór: {name}\n")
        f.write(f"Wzorców: {n_pat}, neurony: {n_neu}, kształt: {shape}\n\n")
        f.write("=== Stabilność ===\n")
        f.write(f"Hebb: {stab['hebb']['n_stable']}/{n_pat} ({stab['hebb']['fraction']*100:.1f}%)\n")
        f.write(f"Oja : {fmt_mean_std(stab['oja']['n_stable_mean'], stab['oja']['n_stable_std'], '.2f')}"
                f" / {n_pat} ({stab['oja']['fraction_mean']*100:.1f}%)\n")
        f.write(f"  raw Oja: {stab['oja']['raw']}\n\n")

        if recovery_hebb:
            f.write("=== Odzyskiwanie (Hebb) ===\n")
            for lvl, s in recovery_hebb.items():
                f.write(f"  noise={lvl:.2f}: overlap={fmt_mean_std(s['overlap_mean'], s['overlap_std'])}  "
                        f"perfect={s['perfect_rate']*100:.1f}%\n")
            f.write("\n=== Odzyskiwanie (Oja) ===\n")
            for lvl, s in recovery_oja.items():
                f.write(f"  noise={lvl:.2f}: overlap={fmt_mean_std(s['overlap_mean'], s['overlap_std'])}  "
                        f"perfect={s['perfect_rate']*100:.1f}%\n")

        if random_info:
            f.write(f"\n=== Losowe wejście ===\n")
            f.write(f"unikalne atraktory: {random_info['unique_attractors']}\n")
            f.write(f"hit training: {random_info['hit_training_rate']*100:.1f}%\n")

    print(f"\n  [wyniki] zapisano → {out_path}")
    return {
        "stability": stab,
        "recovery_hebb": recovery_hebb,
        "recovery_oja": recovery_oja,
        "random": random_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Eksperymenty Hopfield")
    parser.add_argument("datasets", nargs="*", help="Nazwy zbiorów (domyślnie wszystkie)")
    parser.add_argument("--list", action="store_true", help="Pokaż dostępne zbiory")
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--no-random", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Dostępne zbiory:")
        for k, v in DATASETS.items():
            print(f"  {k:20s} → {v['file']}  {v['shape']}")
        return

    names = args.datasets if args.datasets else list(DATASETS.keys())
    # kolejność od najmniejszych
    order = ["small-7x7", "letters-abc-8x12", "animals-14x9", "OCRA-12x30",
             "letters-14x20", "large-25x25", "large-25x25.plus", "large-25x50"]
    names = [n for n in order if n in names] + [n for n in names if n not in order]

    all_results = {}
    for name in names:
        if name not in DATASETS:
            print(f"[!] Nieznany zbiór: {name}")
            continue
        try:
            all_results[name] = run_one_dataset(
                name,
                do_recovery=not args.no_recovery,
                do_random=not args.no_random,
            )
        except Exception as e:
            print(f"[BŁĄD] {name}: {e}")
            import traceback
            traceback.print_exc()

    # zbiorczy skrót
    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PODSUMOWANIE STABILNOŚCI (Hebb / Oja)\n")
        f.write("=" * 60 + "\n")
        for name, res in all_results.items():
            st = res["stability"]
            f.write(f"{name:20s}  Hebb {st['hebb']['n_stable']:2d}  |  "
                    f"Oja {st['oja']['n_stable_mean']:.1f}±{st['oja']['n_stable_std']:.1f}\n")
    print(f"\nZbiorczy raport → {summary_path}")


if __name__ == "__main__":
    main()
