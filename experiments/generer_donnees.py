from pathlib import Path

import numpy as np

from src.synthetic import abondances, axe_spectral, melange, spectres_purs

GRAINE = 0
dossier = Path("data")
dossier.mkdir(exist_ok=True)

axe = axe_spectral()
M = spectres_purs(axe)
A, malade = abondances(taille=64, graine=GRAINE)

for snr in [40, 30, 20, 10]:
    Y, Y_propre, fond = melange(axe, M, A, snr_db=snr, graine=GRAINE)
    chemin = dossier / f"synthetique_snr{snr}.npz"
    np.savez_compressed(chemin, Y=Y, M=M, A=A, malade=malade, axe=axe,
                        snr_db=snr, graine=GRAINE)
    print(f"{chemin} : Y {Y.shape}")