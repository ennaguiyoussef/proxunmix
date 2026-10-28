import matplotlib.pyplot as plt

from src.synthetic import axe_spectral, pic_gaussien, pic_lorentzien

# 1. L'axe spectral et les deux pics (mêmes paramètres)
x = axe_spectral()
g = pic_gaussien(x, centre=1650, largeur=20, amplitude=1.0)
l = pic_lorentzien(x, centre=1650, largeur=20, amplitude=1.0)

# 2. Deux graphiques côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Graphique 1 : échelle normale
ax1.plot(x, g, label="gaussien")
ax1.plot(x, l, label="lorentzien")
ax1.set_xlim(1400, 1900)
ax1.set_title("Échelle linéaire")
ax1.set_xlabel("Nombre d'onde (cm⁻¹)")
ax1.set_ylabel("Intensité")
ax1.legend()

# Graphique 2 : échelle logarithmique (montre les queues)
ax2.plot(x, g, label="gaussien")
ax2.plot(x, l, label="lorentzien")
ax2.set_xlim(1400, 1900)
ax2.set_ylim(1e-4, 2)
ax2.set_yscale("log")
ax2.set_title("Échelle logarithmique")
ax2.set_xlabel("Nombre d'onde (cm⁻¹)")
ax2.legend()

plt.tight_layout()
plt.show()