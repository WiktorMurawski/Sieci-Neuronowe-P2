import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Literal
from matplotlib.widgets import Slider, Button

class HopfieldNetwork:
    def __init__(self, n_neurons: int):
        """
        n_neurons: liczba neuronów (musi być zgodna z długością wektorów wzorców)
        """
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.patterns = None          # zapisane wzorce uczące (do analizy stabilności)
        self.learning_rule = None

    def _hebb_learning(self, patterns: np.ndarray):
        """Klasyczna reguła Hebba (outer product)."""
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)
        self.W /= self.n               # normalizacja (opcjonalna, ale pomaga)

    def _oja_learning(self, patterns: np.ndarray, eta: float = 0.01, epochs: int = 100):
        """
        Reguła Oji dostosowana do sieci Hopfielda.
        Iteracyjnie aktualizujemy wagi, żeby uzyskać lepszą normalizację
        i nieco lepszą pojemność przy skorelowanych wzorcach.
        """
        self.W = np.zeros((self.n, self.n))
        for _ in range(epochs):
            for p in patterns:
                # przewidywanie
                y = self.W @ p
                # reguła Oji: ΔW = η (y p^T - y y^T W)  (wersja macierzowa)
                # uproszczona i stabilniejsza wersja dla Hopfielda:
                self.W += eta * (np.outer(y, p) - np.outer(y, y) @ self.W)
        np.fill_diagonal(self.W, 0)
        # dodatkowo możemy zsymetryzować
        self.W = 0.5 * (self.W + self.W.T)

    def train(self, patterns: List[np.ndarray], rule: Literal["hebb", "oja"] = "hebb",
              oja_eta: float = 0.01, oja_epochs: int = 50):
        """
        patterns: lista wektorów (każdy shape (n,) lub (h, w) – wtedy spłaszczamy)
        """
        # spłaszczenie i bipolaryzacja
        flat = []
        for p in patterns:
            p = np.asarray(p).flatten()
            if p.shape[0] != self.n:
                raise ValueError(f"Wzorzec ma długość {p.shape[0]}, a sieć {self.n}")
            # zakładamy, że wejście jest już ±1 lub 0/1
            p = np.where(p > 0, 1, -1)
            flat.append(p)
        patterns = np.array(flat)
        self.patterns = patterns
        self.learning_rule = rule

        if rule == "hebb":
            self._hebb_learning(patterns)
        elif rule == "oja":
            self._oja_learning(patterns, eta=oja_eta, epochs=oja_epochs)
        else:
            raise ValueError("rule musi być 'hebb' lub 'oja'")

    def energy(self, state: np.ndarray) -> float:
        """Energia Hopfielda: E = -0.5 * s^T W s"""
        s = state.flatten()
        return -0.5 * s @ self.W @ s

    def update(self, state: np.ndarray, mode: Literal["sync", "async"] = "async",
               max_steps: int = 100, track: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Zwraca stan końcowy oraz (opcjonalnie) historię stanów.
        """
        s = state.flatten().copy()
        history = [s.copy()] if track else []

        if mode == "sync":
            for _ in range(max_steps):
                s_new = np.sign(self.W @ s)
                s_new[s_new == 0] = 1          # umowa przy zerze
                if track:
                    history.append(s_new.copy())
                if np.array_equal(s_new, s):
                    break
                s = s_new
        else:  # asynchronous
            for step in range(max_steps):
                order = np.random.permutation(self.n)
                changed = False
                for i in order:
                    h = self.W[i] @ s
                    new_val = 1 if h >= 0 else -1
                    if new_val != s[i]:
                        s[i] = new_val
                        changed = True
                        if track:
                            history.append(s.copy())
                if not changed:
                    break

        return s, history

    def is_stable(self, pattern: np.ndarray) -> bool:
        """Czy dany wzorzec jest stabilnym stanem stałym."""
        p = np.where(np.asarray(pattern).flatten() > 0, 1, -1)
        recovered, _ = self.update(p, mode="sync", max_steps=1)
        return np.array_equal(recovered, p)

    def visualize(self, state: np.ndarray, shape: Tuple[int, int], title: str = "",
                  ax: Optional[plt.Axes] = None):
        """Wyświetla stan jako bitmapę."""
        img = state.reshape(shape)
        if ax is None:
            plt.figure(figsize=(4, 4))
            ax = plt.gca()
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    def animate_convergence(self, initial_state: np.ndarray, shape: Tuple[int, int],
                            mode: Literal["sync", "async"] = "async",
                            max_steps: int = 50, interval: int = 300):
        """Animacja zbiegania do stanu stabilnego."""
        final, history = self.update(initial_state, mode=mode, max_steps=max_steps, track=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(history[0].reshape(shape), cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
        title = ax.set_title(f"Krok 0 / {len(history)-1}")

        def update(frame):
            im.set_data(history[frame].reshape(shape))
            title.set_text(f"Krok {frame} / {len(history)-1}")
            return [im, title]

        ani = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=True, repeat=False)
        plt.show()
        return final, history

    def interactive_convergence(self, initial_state: np.ndarray, shape: Tuple[int, int],
                            mode: Literal["sync", "async"] = "async",
                            max_steps: int = 80):
        """
        Interaktywny podgląd zbiegania.
        """
        final, history = self.update(initial_state, mode=mode, max_steps=max_steps, track=True)
        n_steps = len(history)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.25)          # miejsce na suwak i przyciski

        im = ax.imshow(history[0].reshape(shape), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"Krok 0 / {n_steps-1}   |   Energia: {self.energy(history[0]):.2f}")
        ax.axis("off")

        # Suwak
        ax_slider = plt.axes([0.15, 0.12, 0.7, 0.04])
        slider = Slider(ax_slider, "Krok", 0, n_steps-1, valinit=0, valstep=1)

        # Przyciski
        ax_prev = plt.axes([0.25, 0.03, 0.15, 0.05])
        ax_next = plt.axes([0.60, 0.03, 0.15, 0.05])
        btn_prev = Button(ax_prev, "◀ Poprzedni")
        btn_next = Button(ax_next, "Następny ▶")

        def update_frame(i):
            i = int(i)
            im.set_data(history[i].reshape(shape))
            ax.set_title(f"Krok {i} / {n_steps-1}   |   Energia: {self.energy(history[i]):.2f}")
            fig.canvas.draw_idle()

        def on_slider(val):
            update_frame(val)

        def on_prev(event):
            new_val = max(0, slider.val - 1)
            slider.set_val(new_val)

        def on_next(event):
            new_val = min(n_steps-1, slider.val + 1)
            slider.set_val(new_val)

        slider.on_changed(on_slider)
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

        plt.show()
        return final, history