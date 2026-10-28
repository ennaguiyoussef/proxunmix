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