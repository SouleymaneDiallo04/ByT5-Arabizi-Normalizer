"""Bruits et mesure de evaluation/robustesse_modele.py, sans modele."""

import random

import pandas as pd
import pytest

from evaluation.robustesse_modele import (
    BRUITS, allonger, bruiter, capitale, chiffre, coller, convention, emojier,
    fauter, mesurer, sans_emojis, sans_ponctuation, sans_voyelles)


def t(graine="x"):
    return random.Random(graine)


def test_capitale_et_conventions():
    assert capitale("wach nta mzyan", t()) == "Wach nta mzyan"
    assert convention("ch", "sh")("chrit chi haja", t()) == "shrit shi haja"
    assert convention("kh", "5")("Khouya", t()) == "5ouya"


def test_les_chiffres_nombres_ne_sont_pas_touches():
    sept = chiffre("7", "h")
    assert sept("s7i7 had 7 ayam", t()) == "shih had 7 ayam"
    assert chiffre("9", "q")("9alb 9 dyal", t()) == "qalb 9 dyal"


def test_allonger_repete_la_derniere_voyelle_d_un_mot():
    b = allonger("bzaf", t())
    assert b == "bzaaaaf"


def test_fauter_echange_deux_caracteres_voisins():
    b = fauter("mzyan bzaf", t())
    assert b != "mzyan bzaf"
    assert sorted(b) == sorted("mzyan bzaf")


def test_sans_voyelles_et_coller():
    assert sans_voyelles("mzyan", t()) == "mzyn"
    assert coller("wach nta", t()) == "wachnta"
    # Un mot seul n'a rien a coller.
    assert coller("salam", t()) == "salam"


def test_emojier_ajoute_un_emoji_a_la_fin():
    b = emojier("safi", t())
    assert b.startswith("safi") and sans_emojis(b) == "safi"


def test_transformations_de_comparaison():
    assert sans_emojis("واخا 😂 ❤️") == "واخا"
    assert sans_ponctuation("واش نتا مزيان؟؟؟!") == "واش نتا مزيان"


def test_bruiter_est_reproductible_et_ne_garde_que_ce_qui_change():
    test = pd.DataFrame({"id": ["a", "b"], "transliteration": ["salam", "s7i7 chi haja"]})
    l1, l2 = bruiter(test), bruiter(test)
    assert l1.equals(l2)
    # « salam » n'a ni ch, ni kh, ni chiffre : ces bruits ne la touchent pas.
    bruits_a = set(l1[l1["id"] == "a"]["bruit"])
    assert "ch en sh" not in bruits_a and "7 en h" not in bruits_a
    assert "propre" in bruits_a and "majuscules" in bruits_a
    assert {"ch en sh", "7 en h"} <= set(l1[l1["id"] == "b"]["bruit"])
    assert len(BRUITS) == len({nom for nom, _, _ in BRUITS})


def test_mesurer():
    d = pd.DataFrame({"darija_norm": ["شريت", "واش"],
                      "propre": ["شريت", "واش"],
                      "bruitee": ["شريط", "واش"]})
    m = mesurer(d)
    assert m["cer_propre"] == 0
    assert m["cer_bruite"] == pytest.approx(1 / 7)
    assert (m["pire"], m["mieux"], m["perdues"], m["changees"]) == (1, 0, 1, 1)
