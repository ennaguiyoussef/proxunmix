import numpy as np

from src.solvers import moindres_carres, nnls_pixels
from src.synthetic import abondances, axe_spectral, spectres_purs
from src.solvers import fista, gradient_projete , pas_lipschitz


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
    


def test_fista_sans_bruit_retrouve_A():
    M, A, Y = _sans_bruit()
    A_est, _ = fista(M, Y, n_iter=5000, tol=1e-10)
    assert np.allclose(A_est, A, atol=1e-4)


def test_fista_egal_nnls_quand_lambda_nul():
    M, A, Y = _sans_bruit()
    rng = np.random.default_rng(0)
    Y_bruite = Y + rng.normal(0, 0.05, size=Y.shape)
    A_fista, _ = fista(M, Y_bruite, lam=0.0, n_iter=5000, tol=1e-10)
    assert np.allclose(A_fista, nnls_pixels(M, Y_bruite), atol=1e-4)


def test_fista_respecte_la_borne_theorique():
    # Beck et Teboulle (2009) : F(A_k) - F* <= 2 L ||A_0 - A*||² / (k + 1)²
    M, A, Y = _sans_bruit()
    _, historique = fista(M, Y, n_iter=200, tol=0)
    L = 1.0 / pas_lipschitz(M)
    for k, F_k in enumerate(historique, start=1):
        borne = 2 * L * np.sum(A ** 2) / (k + 1) ** 2
        assert F_k <= borne + 1e-9


def test_lambda_grand_donne_zero():
    M, A, Y = _sans_bruit()
    A_est, _ = fista(M, Y, lam=1e6, n_iter=50)
    assert np.allclose(A_est, 0)    