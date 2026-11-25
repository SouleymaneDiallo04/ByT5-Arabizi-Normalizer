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
