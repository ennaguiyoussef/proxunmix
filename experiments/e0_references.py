import time

import numpy as np

from src.metrics import rmse
from src.solvers import moindres_carres, nnls_pixels

SOLVEURS = [("Moindres carrés", moindres_carres), ("NNLS", nnls_pixels)]

print(f"{'SNR':>5} | {'Méthode':<16} | {'RMSE':>7} | {'% nég.':>6} | {'Temps':>7}")
print("-" * 55)

for snr in [40, 30, 20, 10, 0, -5]:
    d = np.load(f"data/synthetique_snr{snr}.npz")
    Y, M, A_vrai = d["Y"], d["M"], d["A"]

    for nom, solveur in SOLVEURS:
        debut = time.perf_counter()
        A_est = solveur(M, Y)
        duree = time.perf_counter() - debut
        print(f"{snr:>5} | {nom:<16} | {rmse(A_est, A_vrai):>7.4f} | "
              f"{100 * np.mean(A_est < 0):>5.1f}% | {duree:>6.3f}s")