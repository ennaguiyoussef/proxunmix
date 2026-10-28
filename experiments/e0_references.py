import time

import numpy as np

from src.metrics import rmse
from src.solvers import moindres_carres, nnls_pixels

d = np.load("data/synthetique_snr30.npz")
Y, M, A_vrai = d["Y"], d["M"], d["A"]
print(f"Données : Y {Y.shape}, M {M.shape}, A {A_vrai.shape}, SNR {d['snr_db']} dB\n")

for nom, solveur in [("Moindres carrés", moindres_carres), ("NNLS", nnls_pixels)]:
    debut = time.perf_counter()
    A_est = solveur(M, Y)
    duree = time.perf_counter() - debut

    print(nom)
    print(f"  RMSE           : {rmse(A_est, A_vrai):.4f}")
    print(f"  Temps          : {duree:.3f} s")
    print(f"  % négatives    : {100 * np.mean(A_est < 0):.1f} %")
    print(f"  Somme moyenne  : {A_est.sum(axis=0).mean():.3f}\n")