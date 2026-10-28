import numpy as np

from src.prox import projection_positive, projection_simplexe, seuillage_doux


def test_seuillage_doux_a_la_main():
    v = np.array([3.0, -2.0, 0.5, -0.4])
    attendu = np.array([2.0, -1.0, 0.0, 0.0])
    assert np.allclose(seuillage_doux(v, 1.0), attendu)


def test_projection_positive():
    v = np.array([0.9, -0.1, 0.3])
    assert np.allclose(projection_positive(v), [0.9, 0.0, 0.3])


def test_simplexe_cas_a_la_main():
    assert np.allclose(projection_simplexe([0.2, 0.2, 0.2]), [1/3, 1/3, 1/3])
    assert np.allclose(projection_simplexe([2.0, 0.0]), [1.0, 0.0])
    assert np.allclose(projection_simplexe([0.7, 0.3]), [0.7, 0.3])  # déjà sur le simplexe


def test_simplexe_colonnes_valides():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(3, 500))
    X = projection_simplexe(V)
    assert X.shape == (3, 500)
    assert np.all(X >= 0)
    assert np.allclose(X.sum(axis=0), 1.0)


def test_simplexe_est_le_point_le_plus_proche():
    rng = np.random.default_rng(1)
    v = rng.normal(size=3)
    x = projection_simplexe(v)
    # Aucun autre point du simplexe ne doit être plus proche de v
    autres = rng.dirichlet([1, 1, 1], size=2000)
    distances = np.linalg.norm(autres - v, axis=1)
    assert np.linalg.norm(x - v) <= distances.min() + 1e-12