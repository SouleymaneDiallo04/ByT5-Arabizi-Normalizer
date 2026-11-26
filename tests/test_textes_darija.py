"""Fonctions de evaluation/textes_darija.py, sans telechargement."""

from evaluation.textes_darija import lexique, mots, mots_faux, mots_latins, regles


def test_mots_normalises_comme_la_metrique():
    assert mots("واش  كاين؟ 3 مرات!") == ["واش", "كاين", "مرات"]
    assert mots("شْريت") == ["شريت"]          # vocalisation retiree
    assert mots("أنا") != mots("انا")          # hamzas strictes, comme le CER


def test_mots_faux_compte_les_repetitions():
    assert mots_faux("بزاف بزاف ديال", "بزاف ديال") == ["بزاف"]
    assert mots_faux("شريت طوموبيل", "شريت طوموبيل") == []


def test_mots_latins():
    # La ponctuation devient une espace, comme dans cle_latine : l'apostrophe
    # separe l'article.
    assert mots_latins("Wach, 3ndk 7ta l'flouss?") == ["wach", "3ndk", "7ta", "l", "flouss"]


def test_regles():
    assert regles("شني") == "chni"
    assert regles("لعشير") == "l3chir"
    assert regles("وقيلة") == "w9ila"
    # Les deux limites que la mesure chiffre : voyelles breves et ك prononce g
    assert regles("تلفن") == "tlfn"
    assert regles("كبير") == "kbir"


def test_lexique_sur_les_paires_alignees():
    g = lexique([("kanbghik bzaf", "كنبغيك بزاف"),
                 ("bezzaf", "بزاف"),
                 ("trois mots ici", "deux mots")])       # pas d'arabe : ignoree
    assert g["بزاف"] == {"bzaf": 1, "bezzaf": 1}
    assert g["كنبغيك"].most_common(1)[0][0] == "kanbghik"
