import matplotlib.pyplot as plt
import numpy as np

from src.metrics import rmse
from src.solvers import fista, moindres_carres

SNRS = [40, 30, 20, 10, 0, -5]
resultats = {"Moindres carrés": [], "FISTA positif (= NNLS)": [], "FISTA simplexe": []}

print(f"{'SNR':>5} | {'MC':>7} | {'Positif':>7} | {'Simplexe':>8} | {'Gain':>6}")
print("-" * 46)

for snr in SNRS:
    d = np.load(f"data/synthetique_snr{snr}.npz")
    Y, M, A_vrai = d["Y"], d["M"], d["A"]

    r_mc = rmse(moindres_carres(M, Y), A_vrai)
    A_pos, _ = fista(M, Y, n_iter=2000, tol=1e-8, redemarrage=True, contrainte="positif")
    A_sim, _ = fista(M, Y, n_iter=2000, tol=1e-8, redemarrage=True, contrainte="simplexe")
    r_pos, r_sim = rmse(A_pos, A_vrai), rmse(A_sim, A_vrai)

    resultats["Moindres carrés"].append(r_mc)
    resultats["FISTA positif (= NNLS)"].append(r_pos)
    resultats["FISTA simplexe"].append(r_sim)

    gain = 100 * (r_pos - r_sim) / r_pos
    print(f"{snr:>5} | {r_mc:>7.4f} | {r_pos:>7.4f} | {r_sim:>8.4f} | {gain:>5.1f}%")

# Figure : RMSE selon le SNR
plt.figure(figsize=(7, 4))
for nom, valeurs in resultats.items():
    plt.semilogy(SNRS, valeurs, marker="o", label=nom)
plt.gca().invert_xaxis()          # du plus propre (gauche) au plus bruité (droite)
plt.xticks(SNRS)
plt.xlabel("SNR (dB)")
plt.ylabel("RMSE des abondances (log)")
plt.title("Erreur selon le niveau de bruit")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/rmse_snr.png", dpi=150)
plt.show()