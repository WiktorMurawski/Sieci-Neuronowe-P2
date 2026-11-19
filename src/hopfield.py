"""
Sieć Hopfielda

Reguły uczenia: Hebb, Oja
Aktualizacja: synchroniczna / asynchroniczna
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Literal, Sequence
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, n_neurons: int):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.patterns: Optional[np.ndarray] = None
        self.learning_rule: Optional[str] = None
        self.shape: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Uczenie
    # ------------------------------------------------------------------
    def _hebb(self, patterns: np.ndarray):
        """Klasyczna reguła Hebba (outer product + zero diagonal)."""
        self.W = patterns.T @ patterns / self.n
        np.fill_diagonal(self.W, 0.0)

    def _oja(self, patterns: np.ndarray, eta: float = 0.5,
            epochs: int = 3000, tol: float = 1e-7, eps: float = 1e-2):
        """
        Reguła Oji (wersja macierzowa, y = W x), wsadowa:
            dW = eta * (W C - W C W^T W),   C = (1/p) * sum_mu x x^T,  x = xi/sqrt(N)
        Startujemy od W = eps*I; W zbiega do rzutnika na span{xi^mu}.
        """
        X = patterns / np.sqrt(self.n)
        C = X.T @ X / len(patterns)
        W = eps * np.eye(self.n)
        for _ in range(epochs):
            Y = W @ C
            dW = eta * (Y - Y @ W.T @ W)
            W += dW
            if np.abs(dW).max() < tol:
                break
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        self.W = W

    def train(
        self,
        patterns: Sequence[np.ndarray],
        rule: Literal["hebb", "oja"] = "hebb",
        shape: Optional[Tuple[int, int]] = None,
    ):
        flat = []
        for p in patterns:
            p = np.asarray(p, dtype=float).flatten()
            if p.size != self.n:
                raise ValueError(f"Wzorzec ma długość {p.size}, sieć ma {self.n}")
            p = np.where(p > 0, 1.0, -1.0)
            flat.append(p)
        patterns_arr = np.array(flat)
        self.patterns = patterns_arr
        self.learning_rule = rule
        self.shape = shape

        if rule == "hebb":
            self._hebb(patterns_arr)
        elif rule == "oja":
            self._oja(patterns_arr)
        else:
            raise ValueError("rule musi być 'hebb' lub 'oja'")

    # ------------------------------------------------------------------
    # Dynamika
    # ------------------------------------------------------------------
    def energy(self, state: np.ndarray) -> float:
        s = state.flatten()
        return -0.5 * float(s @ self.W @ s)

    def update(
        self,
        state: np.ndarray,
        mode: Literal["sync", "async"] = "async",
        max_steps: int = 100,
        track: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        s = np.where(np.asarray(state).flatten() > 0, 1.0, -1.0).copy()
        history = [s.copy()] if track else []

        if mode == "sync":
            for _ in range(max_steps):
                s_new = np.sign(self.W @ s)
                s_new[s_new == 0] = 1.0
                if track:
                    history.append(s_new.copy())
                if np.array_equal(s_new, s):
                    break
                s = s_new
        else:
            for _ in range(max_steps):
                order = np.random.permutation(self.n)
                changed = False
                for i in order:
                    h = self.W[i] @ s
                    new_val = 1.0 if h >= 0 else -1.0
                    if new_val != s[i]:
                        s[i] = new_val
                        changed = True
                        if track:
                            history.append(s.copy())
                if not changed:
                    break
        return s, history

    def is_stable(self, pattern: np.ndarray, mode: Literal["sync", "async"] = "sync") -> bool:
        p = np.where(np.asarray(pattern).flatten() > 0, 1.0, -1.0)
        recovered, _ = self.update(p, mode=mode, max_steps=5)
        return bool(np.array_equal(recovered, p))

    def stability_report(self) -> dict:
        if self.patterns is None:
            raise RuntimeError("Sieć nie została jeszcze nauczona")
        results = []
        for i, p in enumerate(self.patterns):
            results.append({
                "index": i,
                "stable_sync": self.is_stable(p, mode="sync"),
                "stable_async": self.is_stable(p, mode="async"),
                "energy": self.energy(p),
            })
        n_stable = sum(r["stable_sync"] for r in results)
        return {
            "n_patterns": len(self.patterns),
            "n_stable_sync": n_stable,
            "fraction_stable": n_stable / len(self.patterns),
            "details": results,
            "rule": self.learning_rule,
        }

    def corrupt(self, pattern: np.ndarray, noise_level: float) -> np.ndarray:
        p = np.where(np.asarray(pattern).flatten() > 0, 1.0, -1.0).copy()
        n_flip = int(round(noise_level * self.n))
        if n_flip > 0:
            idx = np.random.choice(self.n, size=n_flip, replace=False)
            p[idx] *= -1
        return p

    # ------------------------------------------------------------------
    # Wizualizacja
    # ------------------------------------------------------------------
    def visualize(self, state: np.ndarray, title: str = "", ax=None):
        if self.shape is None:
            raise ValueError("Nie ustawiono shape - podaj shape przy train()")
        img = np.asarray(state).reshape(self.shape)
        if ax is None:
            plt.figure(figsize=(4, 4))
            ax = plt.gca()
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    def interactive_convergence(
        self,
        initial_state: np.ndarray,
        mode: Literal["sync", "async"] = "async",
        max_steps: int = 80,
    ):
        if self.shape is None:
            raise ValueError("Nie ustawiono shape")
        final, history = self.update(initial_state, mode=mode, max_steps=max_steps, track=True)
        n_steps = len(history)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.25)
        im = ax.imshow(history[0].reshape(self.shape), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"Krok 0 / {n_steps-1}  |  E = {self.energy(history[0]):.1f}")
        ax.axis("off")

        ax_slider = plt.axes([0.15, 0.12, 0.7, 0.04])
        slider = Slider(ax_slider, "Krok", 0, n_steps - 1, valinit=0, valstep=1)
        ax_prev = plt.axes([0.25, 0.03, 0.15, 0.05])
        ax_next = plt.axes([0.60, 0.03, 0.15, 0.05])
        btn_prev = Button(ax_prev, "Poprzedni")
        btn_next = Button(ax_next, "Następny")

        def update_frame(i):
            i = int(i)
            im.set_data(history[i].reshape(self.shape))
            ax.set_title(f"Krok {i} / {n_steps-1}  |  E = {self.energy(history[i]):.1f}")
            fig.canvas.draw_idle()

        slider.on_changed(lambda v: update_frame(v))
        btn_prev.on_clicked(lambda e: slider.set_val(max(0, slider.val - 1)))
        btn_next.on_clicked(lambda e: slider.set_val(min(n_steps - 1, slider.val + 1)))
        plt.show()
        return final, history