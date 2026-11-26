"""Normaliseur.normaliser : une phrase a la fois en int8, par lots en fp32.

Le modele n'est pas charge : decoder est remplace par un faux qui note la
taille de chaque lot recu.
"""

import threading

import pytest

import service.normaliseur as module
from service.normaliseur import Normaliseur


@pytest.fixture
def lots(monkeypatch):
    recus = []

    def faux_decoder(modele, tok, phrases, appareil, taille_lot=16):
        # Le vrai decoder coupe en lots de taille_lot : le faux note leur taille.
        sorties = ([], [], [])
        for i in range(0, len(phrases), taille_lot):
            lot = phrases[i:i + taille_lot]
            recus.append(len(lot))
            sorties[0].extend(f"<{p}>" for p in lot)
            sorties[1].extend([0.5] * len(lot))
            sorties[2].extend([True] * len(lot))
        return sorties

    monkeypatch.setattr(module, "decoder", faux_decoder)
    return recus


def normaliseur(variante):
    n = object.__new__(Normaliseur)
    n.modele = n.tok = None
    n.appareil, n.variante, n._verrou = "cpu", variante, threading.Lock()
    return n


def test_int8_decode_une_phrase_a_la_fois(lots):
    r = normaliseur("int8").normaliser(["a", "b", "c"])
    assert lots == [1, 1, 1]
    assert r == [("<a>", 0.5, True), ("<b>", 0.5, True), ("<c>", 0.5, True)]


def test_fp32_garde_ses_lots(lots):
    r = normaliseur("fp32").normaliser(["a", "b", "c"])
    assert lots == [3]
    assert [s for s, _, _ in r] == ["<a>", "<b>", "<c>"]
