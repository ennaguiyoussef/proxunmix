import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import rmse
from src.solvers import fista, gradient_projete, nnls_pixels

d = np.load("data/synthetique_snr0.npz")
Y, M, A_vrai = d["Y"], d["M"], d["A"]
print(f"Données : Y {Y.shape}, SNR {d['snr_db']} dB\n")

# ---------------------------------------------------------------------------
# 1. Convergence des trois méthodes (λ = 0)
# ---------------------------------------------------------------------------
methodes = [
    ("Gradient projeté", gradient_projete),
    ("FISTA", fista),
    ("FISTA + redémarrage", partial(fista, redemarrage=True)),
]

resultats = {}
print(f"{'Méthode':<22} | {'RMSE':>7} | {'Temps':>8}")
print("-" * 44)
for nom, solveur in methodes:
    debut = time.perf_counter()
    A_est, historique = solveur(M, Y, lam=0.0, n_iter=300, tol=0)
    duree = time.perf_counter() - debut
    resultats[nom] = historique
    print(f"{nom:<22} | {rmse(A_est, A_vrai):>7.4f} | {duree:>7.3f}s")

debut = time.perf_counter()
A_nnls = nnls_pixels(M, Y)
duree = time.perf_counter() - debut
print(f"{'NNLS (scipy)':<22} | {rmse(A_nnls, A_vrai):>7.4f} | {duree:>7.3f}s")

# Courbes de convergence : F(A_k) - F*, avec F* = plus petite valeur atteinte
F_etoile = min(min(h) for h in resultats.values())
plt.figure(figsize=(8, 4))
for nom, historique in resultats.items():
    plt.semilogy(np.array(historique) - F_etoile + 1e-12, label=nom)
plt.xlabel("Itération k")
plt.ylabel("F(A_k) − F*  (échelle log)")
plt.title("Convergence à SNR 0 dB (λ = 0)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/convergence.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 2. Balayage de λ avec FISTA + redémarrage
# ---------------------------------------------------------------------------
print("\nBalayage de λ (FISTA + redémarrage) :")
print(f"{'λ':>8} | {'RMSE':>7}")
print("-" * 18)
for lam in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
    A_est, _ = fista(M, Y, lam=lam, n_iter=1000, redemarrage=True)
    print(f"{lam:>8} | {rmse(A_est, A_vrai):>7.4f}")