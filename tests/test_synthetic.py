import numpy as np

from src.synthetic import axe_spectral, pic_gaussien, pic_lorentzien
from src.synthetic import spectres_purs
from src.synthetic import abondances
from src.synthetic import melange


def test_axe_spectral():
    axe = axe_spectral()
    assert len(axe) == 1024
    assert axe[0] == 500
    assert axe[-1] == 3200


def test_pic_gaussien_maximum():
    axe = axe_spectral()
    pic = pic_gaussien(axe, centre=1650, largeur=20, amplitude=2.0)
    pas = axe[1] - axe[0]

    assert np.isclose(np.max(pic), 2.0, atol=1e-2)
    assert np.isclose(axe[np.argmax(pic)], 1650, atol=pas / 2)


def test_pic_lorentzien_maximum():
    axe = axe_spectral()
    pic = pic_lorentzien(axe, centre=1650, largeur=20, amplitude=2.0)
    pas = axe[1] - axe[0]

    assert np.isclose(np.max(pic), 2.0, atol=1e-2)
    assert np.isclose(axe[np.argmax(pic)], 1650, atol=pas / 2)


def test_lorentzien_demi_hauteur():
    # À une demi-largeur du centre, le pic lorentzien vaut la moitié de l'amplitude
    axe = np.array([1650.0, 1670.0])
    pic = pic_lorentzien(axe, centre=1650, largeur=20, amplitude=2.0)
    assert np.isclose(pic[1], 1.0)
    
    
def test_spectres_purs_forme_et_normalisation():
    axe = axe_spectral()
    M = spectres_purs(axe)
    assert M.shape == (1024, 3)
    assert np.all(M >= 0)
    assert np.allclose(M.max(axis=0), 1.0)


def test_zone_muette_un_seul_composant():
    axe = axe_spectral()
    M = spectres_purs(axe)
    zone = (axe >= 2100) & (axe <= 2200)
    max_zone = M[zone].max(axis=0)   # max de chaque composant dans la zone muette
    assert max_zone[1] > 0.03        # le lipide a un signal visible
    assert max_zone[0] < 0.01        # la protéine presque rien
    assert max_zone[2] < 0.01        # le collagène presque rien    
    


def test_abondances_simplexe():
    A, malade = abondances(taille=32, graine=1)
    assert A.shape == (3, 32 * 32)
    assert np.all(A >= 0)
    assert np.allclose(A.sum(axis=0), 1.0)


def test_abondances_reproductibles():
    A1, _ = abondances(graine=42)
    A2, _ = abondances(graine=42)
    assert np.array_equal(A1, A2)


def test_regions_saine_et_malade():
    A, malade = abondances(graine=0)
    proteine_sain = A[0, ~malade].mean()
    proteine_malade = A[0, malade].mean()
    assert proteine_sain > proteine_malade
    assert np.isclose(proteine_sain, 0.8, atol=0.02)
    



def _donnees(snr_db=30, amplitude_fond=0.0, graine=0):
    axe = axe_spectral()
    M = spectres_purs(axe)
    A, _ = abondances(taille=16, graine=graine)
    return axe, M, A, melange(axe, M, A, snr_db, amplitude_fond, graine)


def test_melange_forme():
    axe, M, A, (Y, Y_propre, fond) = _donnees()
    assert Y.shape == (1024, 16 * 16)
    assert np.allclose(Y_propre, M @ A)
    assert np.all(fond == 0)   # amplitude_fond = 0 par défaut


def test_snr_respecte():
    _, _, _, (Y, Y_propre, fond) = _donnees(snr_db=20)
    bruit = Y - Y_propre - fond
    snr_mesure = 10 * np.log10(np.mean(Y_propre ** 2) / np.mean(bruit ** 2))
    assert np.isclose(snr_mesure, 20, atol=0.2)


def test_fond_positif_quand_active():
    _, _, _, (_, _, fond) = _donnees(amplitude_fond=0.3)
    assert np.all(fond >= 0)
    assert fond.max() > 0    