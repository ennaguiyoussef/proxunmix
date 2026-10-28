import numpy as np

from src.solvers import moindres_carres, nnls_pixels
from src.synthetic import abondances, axe_spectral, spectres_purs


def _sans_bruit():
    axe = axe_spectral()
    M = spectres_purs(axe)
    A, _ = abondances(taille=8, graine=0)
    return M, A, M @ A


def test_moindres_carres_sans_bruit():
    M, A, Y = _sans_bruit()
    assert np.allclose(moindres_carres(M, Y), A, atol=1e-8)


def test_nnls_sans_bruit():
    M, A, Y = _sans_bruit()
    assert np.allclose(nnls_pixels(M, Y), A, atol=1e-6)


def test_nnls_positif():
    M, A, Y = _sans_bruit()
    rng = np.random.default_rng(0)
    Y_bruite = Y + rng.normal(0, 0.1, size=Y.shape)
    assert np.all(nnls_pixels(M, Y_bruite) >= 0)