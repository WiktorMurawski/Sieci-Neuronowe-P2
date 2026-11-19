#!/usr/bin/env python3
"""
Orkiestrator eksperymentów

Użycie:
    python src/run_all.py
    python src/run_all.py --only stability recovery
    python src/run_all.py --list
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import DATASETS, list_datasets, load_dataset, project_root
from experiments.previews import save_previews
from experiments.stability import run_stability
from experiments.recovery import run_recovery
from experiments.compare_sets import run_compare_25x25_vs_25x50, run_compare_similar_vs_diverse
from experiments.random_input import run_random_input


def generate_all_previews():
    print("\n=== Generowanie podglądów wzorców ===")
    for name in DATASETS:
        patterns, shape = load_dataset(name)
        dest = save_previews(name, patterns, shape)
        print(f"  {name:20s} -> {dest}")


def main():
    parser = argparse.ArgumentParser(description="Eksperymenty Hopfield – punkty 1–6")
    parser.add_argument("--list", action="store_true", help="Pokaż dostępne zbiory")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["previews", "stability", "recovery", "compare3", "compare5", "random"],
        help="Uruchom tylko wybrane eksperymenty",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Liczba ziaren (domyślnie 5)")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    selected = set(args.only) if args.only else {
        "previews", "stability", "recovery", "compare3", "compare5", "random"
    }

    print("Projekt:", project_root())
    print("Wyniki będą w:", project_root() / "results")
    print()

    if "previews" in selected:
        generate_all_previews()

    if "stability" in selected:
        print("\n=== PUNKT 1+2: Stabilność (Hebb vs Oja) ===")
        run_stability(n_seeds=args.seeds)

    if "recovery" in selected:
        print("\n=== PUNKT 4: Odzyskiwanie z szumu ===")
        run_recovery(n_seeds=args.seeds)

    if "compare3" in selected:
        print("\n=== PUNKT 3: 25×25 vs 25×50 ===")
        run_compare_25x25_vs_25x50(n_seeds=args.seeds)

    if "compare5" in selected:
        print("\n=== PUNKT 5: podobne vs zróżnicowane ===")
        run_compare_similar_vs_diverse(n_seeds=args.seeds)

    if "random" in selected:
        print("\n=== PUNKT 6: Losowe wejście ===")
        run_random_input(n_seeds=args.seeds)

    print("\n" + "=" * 50)
    print("Raporty zapisano do katalogu results/:")
    results_dir = project_root() / "results"
    if results_dir.exists():
        for f in sorted(results_dir.glob("*.txt")):
            print(f"  {f.name}")


if __name__ == "__main__":
    main()