import numpy as np
from scipy.optimize import nnls
from src.prox import projection_positive


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


def pas_lipschitz(M):
    """Pas γ = 1/L avec L = ||M||₂², qui garantit la convergence."""
    return 1.0 / np.linalg.norm(M, 2) ** 2


def objectif(MtM, MtY, normeY2, A, lam):
    """F(A) = ½||Y - MA||² + λ Σ|A|, calculé sans former M @ A (plus rapide)."""
    terme_donnees = 0.5 * np.sum(A * (MtM @ A)) - np.sum(A * MtY) + 0.5 * normeY2
    return terme_donnees + lam * np.sum(np.abs(A))


def _variation_relative(A, A_ancien):
    return np.linalg.norm(A - A_ancien) / max(np.linalg.norm(A_ancien), 1e-12)


def gradient_projete(M, Y, lam=0.0, n_iter=500, tol=1e-6):
    """Gradient proximal (ISTA) pour min ½||Y - MA||² + λ||A||₁ avec A ≥ 0."""
    gamma = pas_lipschitz(M)
    MtM, MtY, normeY2 = M.T @ M, M.T @ Y, np.sum(Y ** 2)
    A = np.zeros((M.shape[1], Y.shape[1]))
    historique = []

    for _ in range(n_iter):
        A_ancien = A
        gradient = MtM @ A - MtY
        A = projection_positive(A - gamma * gradient - gamma * lam)
        historique.append(objectif(MtM, MtY, normeY2, A, lam))
        if _variation_relative(A, A_ancien) < tol:
            break
    return A, historique


def fista(M, Y, lam=0.0, n_iter=500, tol=1e-6, redemarrage=False):
    """FISTA (Beck et Teboulle, 2009), avec redémarrage adaptatif en option
    (O'Donoghue et Candès, 2015)."""
    gamma = pas_lipschitz(M)
    MtM, MtY, normeY2 = M.T @ M, M.T @ Y, np.sum(Y ** 2)
    A = np.zeros((M.shape[1], Y.shape[1]))
    Z = A.copy()
    t = 1.0
    historique = []

    for _ in range(n_iter):
        A_ancien = A
        gradient = MtM @ Z - MtY
        A = projection_positive(Z - gamma * gradient - gamma * lam)
        historique.append(objectif(MtM, MtY, normeY2, A, lam))

        # Redémarrage adaptatif : si l'objectif remonte, on annule l'élan
        if redemarrage and len(historique) > 1 and historique[-1] > historique[-2]:
            t = 1.0
            Z = A.copy()
        else:
            t_nouveau = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            Z = A + ((t - 1) / t_nouveau) * (A - A_ancien)
            t = t_nouveau

        if _variation_relative(A, A_ancien) < tol:
            break
    return A, historique