import numpy as np


def rmse(A_estime, A_vrai):
    """Erreur quadratique moyenne entre abondances estimées et vraies."""
    return np.sqrt(np.mean((A_estime - A_vrai) ** 2))