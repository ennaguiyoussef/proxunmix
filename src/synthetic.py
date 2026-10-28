import numpy as np


def axe_spectral(n=1024, debut=500, fin=3200):
    """Renvoie n nombres d'onde régulièrement espacés entre debut et fin (cm⁻¹)."""
    return np.linspace(debut, fin, n)


def pic_gaussien(axe, centre, largeur, amplitude):
    """Pic gaussien centré en `centre`, de largeur `largeur` et de hauteur `amplitude`."""
    return amplitude * np.exp(-((axe - centre) ** 2) / (2 * largeur ** 2))


def pic_lorentzien(axe, centre, largeur, amplitude):
    """Pic lorentzien centré en `centre`, de demi-largeur `largeur` et de hauteur `amplitude`."""
    return amplitude / (1 + ((axe - centre) / largeur) ** 2)


# Noms des composants, dans l'ordre des colonnes de M
COMPOSANTS = ["protéine", "lipide", "collagène"]

# Pour chaque composant : liste de pics (centre, largeur, amplitude)
PICS = {
    "protéine": [(1003, 6, 0.6), (1650, 20, 0.9), (2930, 25, 1.0)],
    "lipide": [(1300, 15, 0.5), (1440, 15, 0.7), (2850, 15, 1.0),
               (2150, 15, 0.05)],  # faible pic dans la zone muette (sonde C–D)
    "collagène": [(855, 10, 0.6), (940, 12, 0.5), (1665, 20, 1.0)],
}


def spectre_depuis_pics(axe, pics):
    """Somme de pics lorentziens. `pics` est une liste de (centre, largeur, amplitude)."""
    spectre = np.zeros_like(axe)
    for centre, largeur, amplitude in pics:
        spectre += pic_lorentzien(axe, centre, largeur, amplitude)
    return spectre


def spectres_purs(axe):
    """Renvoie la matrice M (len(axe) × 3) : une colonne normalisée par composant."""
    colonnes = [spectre_depuis_pics(axe, PICS[nom]) for nom in COMPOSANTS]
    M = np.column_stack(colonnes)
    M = M / M.max(axis=0)  # chaque colonne a un maximum égal à 1
    return M

def abondances(taille=64, alpha_sain=(8, 1, 1), alpha_malade=(3, 3, 3), graine=0):
    """Abondances A (3 × taille²) : un disque « malade » au centre, le reste « sain ».

    Renvoie aussi le masque booléen (taille²,) des pixels malades.
    """
    rng = np.random.default_rng(graine)

    # Coordonnées (ligne, colonne) de chaque pixel
    lignes, colonnes = np.indices((taille, taille))

    # Disque malade au centre, de rayon taille / 4
    centre = taille / 2
    rayon = taille / 4
    malade = (lignes - centre) ** 2 + (colonnes - centre) ** 2 < rayon ** 2
    malade = malade.ravel()  # image 2D → vecteur 1D de taille²

    # Tirage Dirichlet séparé pour chaque région
    A = np.empty((3, taille * taille))
    A[:, ~malade] = rng.dirichlet(alpha_sain, size=np.sum(~malade)).T
    A[:, malade] = rng.dirichlet(alpha_malade, size=np.sum(malade)).T
    return A, malade