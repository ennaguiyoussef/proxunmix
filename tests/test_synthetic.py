import numpy as np

from src.synthetic import axe_spectral, pic_gaussien, pic_lorentzien


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