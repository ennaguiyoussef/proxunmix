import matplotlib.pyplot as plt

from src.synthetic import abondances, axe_spectral, melange, spectres_purs

axe = axe_spectral()
M = spectres_purs(axe)
A, malade = abondances(taille=64, graine=0)
Y, Y_propre, fond = melange(axe, M, A, snr_db=20, amplitude_fond=0.3, graine=0)

p = 0  # indice d'un pixel (le premier, dans la région saine)

plt.figure(figsize=(10, 4))
plt.plot(axe, Y[:, p], color="lightgrey", label="propre + fond + bruit (SNR 20 dB)")
plt.plot(axe, Y_propre[:, p] + fond[:, p], label="propre + fond")
plt.plot(axe, Y_propre[:, p], label="propre (M @ A)")
plt.xlabel("Nombre d'onde (cm⁻¹)")
plt.ylabel("Intensité")
plt.title(f"Pixel {p} : abondances vraies = {A[:, p].round(2)}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/spectre_pixel.png", dpi=150)
plt.show()