import matplotlib.pyplot as plt

from src.synthetic import COMPOSANTS, abondances

taille = 64
A, malade = abondances(taille=taille, graine=0)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for k, nom in enumerate(COMPOSANTS):
    carte = A[k].reshape(taille, taille)   # ligne k de A → image 64 × 64
    im = axes[k].imshow(carte, vmin=0, vmax=1, cmap="viridis")
    axes[k].set_title(f"Abondance : {nom}")
    axes[k].axis("off")

fig.colorbar(im, ax=axes, shrink=0.8, label="Proportion")
plt.savefig("figures/abondances_vraies.png", dpi=150, bbox_inches="tight")
plt.show()