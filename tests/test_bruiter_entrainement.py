"""Copies bruitees de donnees/bruiter_entrainement.py."""

import pandas as pd

from donnees.bruiter_entrainement import bruiter


def train():
    return pd.DataFrame({
        "id": [f"t{i}" for i in range(6)],
        "transliteration": ["wach nta mzyan", "salam", "chrit l7lib mn l7anout",
                            "bslama 3likom", "a", "kifach daba"],
        "darija_norm": ["واش نتا مزيان", "سلام", "شريت الحليب من الحانوت",
                        "بسلامة عليكم", "ا", "كيفاش دابا"],
        "source": "x",
    })


def test_copies_bruitees_meme_cible_et_reproductibles():
    c1, _ = bruiter(train(), part=1.0)
    c2, _ = bruiter(train(), part=1.0)
    assert c1.equals(c2)
    t = train().set_index("id")
    for x in c1.itertuples():
        origine = x.id.split("_", 1)[1]
        assert x.darija_norm == t.loc[origine, "darija_norm"]
        assert x.transliteration != t.loc[origine, "transliteration"]
        assert x.source.startswith("bruit:")
    # « a » : aucun bruit ne change une seule lettre, elle n'est pas copiee.
    assert not c1["id"].str.endswith("_t4").any()


def test_une_copie_qui_tombe_sur_le_test_est_ecartee():
    copies, _ = bruiter(train(), part=1.0)
    test = pd.DataFrame({"transliteration": copies["transliteration"].head(1)})
    copies2, ecartees = bruiter(train(), part=1.0, test=test)
    assert ecartees == 1 and len(copies2) == len(copies) - 1
