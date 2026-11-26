"""Calculs de evaluation/apparier.py, sur des donnees fabriquees."""

import pandas as pd
import pytest

from evaluation.apparier import a_du_latin, asymetries, comparer, signe


def test_signe():
    assert signe(0, 0) == 1.0
    assert signe(3, 3) == 1.0
    # Cinq phrases sur cinq du meme cote : 2 / 2^5.
    assert signe(5, 0) == pytest.approx(0.0625)
    assert signe(0, 5) == pytest.approx(0.0625)
    # 63 contre 54 : le journal du 21 novembre donne 0,46.
    assert signe(63, 54) == pytest.approx(0.46, abs=0.005)


def test_a_du_latin():
    assert a_du_latin("درت une bonne préparation")
    assert not a_du_latin("درت واحد التحضير مزيان")
    # Les chiffres ne sont pas du latin.
    assert not a_du_latin("جابت 18/20")


def test_comparer():
    d = pd.DataFrame({
        "darija_norm": ["شريت", "واش", "كنقرا"],
        "avant": ["شريت", "واخ", "كنقرا"],
        "apres": ["شريت", "واش", "كنقرا l'informatique"],
    })
    c = comparer(d)
    assert c["exactes"] == (2, 2)
    assert (c["gagnees"], c["perdues"]) == (1, 1)
    assert (c["mieux"], c["pire"]) == (1, 1)
    assert c["latin"] == (0, 1)


def test_asymetries():
    # Une espace en trop, un ا manquant.
    c = asymetries(["فالدار", "باش"], ["ف الدار", "بش"])
    assert c[(" ", "en trop")] == 1
    assert c[("ا", "manquant")] == 1
    assert c[("ل", "en trop")] == 0
