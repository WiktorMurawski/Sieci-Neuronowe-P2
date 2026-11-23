#!/usr/bin/env python3
"""
images_to_patterns.py
Konwertuje zdjęcia z katalogu na wzorce ±1 i zapisuje je do pliku .csv
(jeden wzorzec = jeden wiersz, spłaszczenie wierszami).

Przykład:
    python images_to_patterns.py cats/ cats-200x300.csv --width 200 --height 300
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_pattern(path, width, height, threshold):
    img = Image.open(path).convert("L")                      # skala szarości
    img = ImageOps.fit(img, (width, height), Image.LANCZOS)  # kadr do proporcji + skalowanie
    img = ImageOps.autocontrast(img)                         # rozciągnięcie kontrastu
    a = np.asarray(img, dtype=float)
    thr = np.median(a) if threshold == "median" else a.mean()
    return np.where(a < thr, 1, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_csv")
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--threshold", choices=["median", "mean"], default="median")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in EXT)
    if not files:
        raise SystemExit("Brak obrazów w katalogu.")

    pats = [to_pattern(f, args.width, args.height, args.threshold) for f in files]
    P = np.array([p.flatten() for p in pats])              # (p, height*width)
    np.savetxt(args.output_csv, P, fmt="%d", delimiter=",")

    print(f"Zapisano {len(files)} wzorców, N={P.shape[1]} ({args.height} wierszy x {args.width} kolumn)")
    print(f"Dla data.py: shape = ({args.height}, {args.width})")
    C = P @ P.T / P.shape[1]
    iu = np.triu_indices(len(P), 1)
    if len(P) > 1:
        print(f"Średnia |korelacja| między wzorcami: {np.abs(C[iu]).mean():.2f}")


if __name__ == "__main__":
    main()
