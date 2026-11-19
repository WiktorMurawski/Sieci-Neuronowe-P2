import numpy as np
from hopfield import HopfieldNetwork

# Przykładowe małe wzorce 5x5
def make_pattern(shape, positions):
    p = -np.ones(shape)
    for r, c in positions:
        p[r, c] = 1
    return p

patterns = [
    make_pattern((5, 5), [(1,1),(1,2),(1,3),(2,2),(3,1),(3,2),(3,3)]),  # "O"
    make_pattern((5, 5), [(0,2),(1,1),(1,2),(1,3),(2,2),(3,2),(4,2)]),  # "T"
    make_pattern((5, 5), [(0,0),(0,4),(1,1),(1,3),(2,2),(3,1),(3,3),(4,0),(4,4)]),  # "X"
]

net = HopfieldNetwork(n_neurons=25)
net.train(patterns, rule="hebb")

# Sprawdzenie stabilności
for i, p in enumerate(patterns):
    print(f"Wzorzec {i} stabilny? {net.is_stable(p)}")

# Test odzyskiwania z zaszumionego wejścia
noisy = patterns[0].copy()
noise_idx = np.random.choice(25, size=6, replace=False)
noisy.flat[noise_idx] *= -1

final, history = net.animate_convergence(noisy, shape=(5, 5), mode="async")
print("Energia końcowa:", net.energy(final))

final, history = net.interactive_convergence(noisy, shape=(5, 5), mode="async")
print("Energia końcowa:", net.energy(final))