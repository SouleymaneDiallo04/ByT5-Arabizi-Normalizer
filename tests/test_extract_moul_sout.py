"""Extraction de moul_sout, sur un echantillon fabrique : ni reseau ni jeton."""

import pandas as pd

from donnees.extraction.extract_moul_sout import extraire


def brut():
    return pd.DataFrame({
        "sentence_id": [7, 7, 8, 9, 10],
        "latin_text": ["lmo5alef", "lmokhalif", "chwiya", "  ", None],
        "arabic_text": ["المخالف", "المخالف", "شوية", "والو", "والو"],
    })


def test_colonnes_du_corpus():
    out = extraire(brut())
    assert list(out.columns) == ["id", "darija", "transliteration", "source"]
    assert set(out["source"]) == {"atlasia/moul_sout_transliteration_71k"}


def test_les_deux_arabizi_d_une_phrase_sont_gardes_sous_deux_identifiants():
    out = extraire(brut())
    assert out["id"].tolist() == ["moul-sout_00007_0", "moul-sout_00007_1",
                                  "moul-sout_00008_0"]
    assert out["id"].is_unique


def test_seuls_les_cotes_vides_sont_ecartes():
    out = extraire(brut())
    assert out["transliteration"].tolist() == ["lmo5alef", "lmokhalif", "chwiya"]


def avec_latin():
    return pd.DataFrame({
        "sentence_id": [3, 3, 4, 5, 6],
        "latin_text": ["donc safi", "donc safi", "chwiya", "l'électrique", "ｍerci"],
        "arabic_text": ["donc صافي", "donc صافي", "شوية", "الكهرباء é", "ｍerci"],
    })


def test_le_latin_cote_arabe_est_ecarte_par_defaut():
    out = extraire(avec_latin())
    # Accents seuls et pleine chasse comptent aussi pour du latin.
    assert out["darija"].tolist() == ["شوية"]
    assert out["id"].tolist() == ["moul-sout_00004_0"]


def test_avec_latin_garde_tout_et_les_identifiants_ne_changent_pas():
    tout = extraire(avec_latin(), sans_latin=False)
    assert len(tout) == 5
    assert tout["id"].tolist()[:2] == ["moul-sout_00003_0", "moul-sout_00003_1"]
    sans = extraire(avec_latin())
    assert set(sans["id"]) <= set(tout["id"])
