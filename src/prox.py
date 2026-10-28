import numpy as np


def seuillage_doux(v, seuil):
    """Proximal de seuil * ||x||_1 : signe(v) * max(|v| - seuil, 0)."""
    return np.sign(v) * np.maximum(np.abs(v) - seuil, 0.0)


def projection_positive(v):
    """Projection sur l'orthant positif {x >= 0}."""
    return np.maximum(v, 0.0)


def projection_simplexe(V):
    """Projection de chaque colonne de V sur le simplexe {x >= 0, somme(x) = 1}.

    Algorithme par tri (Held et al. 1974 ; Condat 2016). Accepte un vecteur 1D
    ou une matrice (une colonne = un vecteur à projeter).
    """
    V = np.asarray(V, dtype=float)
    est_1d = V.ndim == 1
    if est_1d:
        V = V[:, None]

    n, m = V.shape
    U = -np.sort(-V, axis=0)                     # chaque colonne triée par ordre décroissant
    cumul = np.cumsum(U, axis=0) - 1.0           # sommes cumulées moins 1
    indices = np.arange(1, n + 1)[:, None]       # 1, 2, ..., n (en colonne)
    condition = U - cumul / indices > 0          # vrai pour les rho premières valeurs
    rho = condition.sum(axis=0)                  # nombre de valeurs qui resteront > 0
    theta = cumul[rho - 1, np.arange(m)] / rho   # seuil de chaque colonne
    X = np.maximum(V - theta, 0.0)

    return X[:, 0] if est_1d else X