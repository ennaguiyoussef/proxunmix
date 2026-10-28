import numpy as np
from scipy.optimize import nnls


def moindres_carres(M, Y):
    """Moindres carrés sans contrainte : résout min ||Y - MA||² pour tous les pixels."""
    A, *_ = np.linalg.lstsq(M, Y, rcond=None)
    return A


def nnls_pixels(M, Y):
    """Moindres carrés avec A >= 0, résolus pixel par pixel avec scipy."""
    n_composants = M.shape[1]
    n_pixels = Y.shape[1]
    A = np.zeros((n_composants, n_pixels))
    for p in range(n_pixels):
        A[:, p], _ = nnls(M, Y[:, p])
    return A