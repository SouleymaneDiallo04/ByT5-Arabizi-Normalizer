"""Regles d'ajout d'une source a l'entrainement, sur un petit corpus construit."""

import pandas as pd

from donnees.ajouter_source import integrer
from donnees.make_splits import cle_arabe, cle_latine, normaliser_cible

COLONNES = ["id", "darija", "transliteration", "source", "key_ar", "key_la",
            "conflit", "suspect", "darija_norm"]


def bloc(paires, prefixe):
    return pd.DataFrame([{
        "id": f"{prefixe}{i}", "darija": ar, "transliteration": la, "source": prefixe,
        "key_ar": cle_arabe(ar), "key_la": cle_latine(la), "conflit": False,
        "suspect": False, "darija_norm": normaliser_cible(ar),
    } for i, (ar, la) in enumerate(paires)], columns=COLONNES)


TRAIN = bloc([("شريت طوموبيلا", "chrit tomobila"), ("فين غادي", "fin ghadi")], "train")
DEV = bloc([("كنبغيك بزاف", "kanbghik bzaf")], "dev")
TEST = bloc([("واش نتا مزيان", "wach nta mzyan")], "test")


def candidat(paires):
    return bloc(paires, "new")[["id", "darija", "transliteration", "source"]]


def test_une_paire_inedite_est_ajoutee_avec_sa_cible_normalisee():
    n, bilan = integrer(candidat([("علاش مشيتي?", "3lach mchiti")]), TRAIN, DEV, TEST)
    assert bilan["ajoutees"] == 1
    assert n.iloc[0]["darija_norm"] == "علاش مشيتي؟"
    assert list(n.columns) == list(TRAIN.columns)


def test_fuite_vers_le_test_par_l_arabe_ou_par_l_arabizi():
    n, bilan = integrer(candidat([("واش نتا مزيان", "wach nta mzyan bzaf"),
                                  ("جملة اخرى طويلة", "WACH nta, mzyan")]), TRAIN, DEV, TEST)
    assert len(n) == 0 and bilan["fuites vers le test"] == 2


def test_fuite_vers_le_dev_par_l_arabe_ou_par_l_arabizi():
    n, bilan = integrer(candidat([("كنبغيك بزاف", "kanbghik bzzaf"),
                                  ("جملة مختلفة هنا", "kanbghik bzaf")]), TRAIN, DEV, TEST)
    assert len(n) == 0 and bilan["fuites vers le dev"] == 2


def test_paire_deja_presente_mais_nouvelle_graphie_gardee():
    n, bilan = integrer(candidat([("شريت طوموبيلا!", "Chrit tomobila"),
                                  ("شريت طوموبيلا", "chrit tomobilaa")]), TRAIN, DEV, TEST)
    assert bilan["deja dans l'entrainement"] == 1
    assert n["transliteration"].tolist() == ["chrit tomobilaa"]


def test_doublon_interne_garde_la_copie_vocalisee():
    n, bilan = integrer(candidat([("علاش", "3lach"), ("عْلاش", "3lach")]), TRAIN, DEV, TEST)
    assert bilan["doublons internes"] == 1 and n.iloc[0]["darija"] == "عْلاش"
    assert n.iloc[0]["darija_norm"] == "علاش"


def test_conflit_marque_et_conserve():
    n, _ = integrer(candidat([("مكان اخر", "fin ghadi daba"), ("فين غادي دابا", "fin ghadi daba")]),
                    TRAIN, DEV, TEST)
    assert len(n) == 2 and n["conflit"].all()


def test_le_bilan_rend_compte_de_chaque_ligne():
    paires = [("علاش مشيتي", "3lach mchiti"), ("واش نتا مزيان", "autre"),
              ("كنبغيك بزاف", "autre2"), ("شريت طوموبيلا", "chrit tomobila"),
              ("علاش مشيتي", "3lach mchiti"), ("...", "...")]
    n, b = integrer(candidat(paires), TRAIN, DEV, TEST)
    ecartees = sum(v for k, v in b.items() if k not in ("lues", "ajoutees"))
    assert ecartees + b["ajoutees"] == b["lues"] == len(paires) and len(n) == 1
