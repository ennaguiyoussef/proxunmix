import matplotlib.pyplot as plt

from src.synthetic import COMPOSANTS, axe_spectral, spectres_purs

axe = axe_spectral()
M = spectres_purs(axe)

fig, ax = plt.subplots(figsize=(10, 4))
for k, nom in enumerate(COMPOSANTS):
    ax.plot(axe, M[:, k], label=nom)

# Zone muette en gris
ax.axvspan(1800, 2700, color="grey", alpha=0.15, label="zone muette")

ax.set_xlabel("Nombre d'onde (cm⁻¹)")
ax.set_ylabel("Intensité normalisée")
ax.set_title("Spectres purs simulés (colonnes de M)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/spectres_purs.png", dpi=150)
plt.show()