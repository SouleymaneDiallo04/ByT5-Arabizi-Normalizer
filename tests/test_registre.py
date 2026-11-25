"""Tests du registre, sur SQLite : ni serveur Postgres ni modele."""

import pytest
from sqlalchemy import text

from registre.depot import Registre


def ligne(entree, confiance, sortie=None):
    return {"entree": entree, "entree_propre": entree.strip(),
            "sortie": sortie or f"<{entree.strip()}>", "confiance": confiance,
            "confiance_brute": 0.95, "complete": True, "revision": "rev-test"}


@pytest.fixture
def registre(tmp_path):
    r = Registre(f"sqlite:///{(tmp_path / 'registre.db').as_posix()}")
    r.creer_tables()
    return r


def test_enregistrer_rend_les_id_dans_l_ordre(registre):
    ids = registre.enregistrer([ligne("a", 0.5), ligne("b", 0.2), ligne("c", 0.9)])
    assert len(set(ids)) == 3
    assert [registre.prediction(i)["entree"] for i in ids] == ["a", "b", "c"]
    assert {registre.prediction(i)["statut"] for i in ids} == {"a_valider"}
    assert registre.prediction(ids[0])["derniere"] is None


def test_creer_tables_deux_fois(registre):
    registre.creer_tables()


def test_a_valider_trie_par_confiance_croissante(registre):
    ids = registre.enregistrer([ligne("a", 0.5), ligne("b", 0.2), ligne("c", 0.9)])
    registre.valider(ids[1], "confirmee", "Awa")
    lignes, total = registre.a_valider()
    assert [x["entree"] for x in lignes] == ["a", "c"] and total == 2
    assert registre.premiere_a_valider() == ids[0]


def test_a_valider_par_pages(registre):
    registre.enregistrer([ligne(f"p{i:02d}", i / 100) for i in range(25)])
    page1, total = registre.a_valider(1, 20)
    page2, _ = registre.a_valider(2, 20)
    assert total == 25 and len(page1) == 20 and len(page2) == 5
    assert page2[0]["entree"] == "p20"


def test_rien_a_valider(registre):
    assert registre.a_valider() == ([], 0)
    assert registre.premiere_a_valider() is None


def test_valider_met_le_statut_a_jour_et_garde_l_historique(registre):
    [i] = registre.enregistrer([ligne("chrit", 0.3)])
    registre.valider(i, "corrigee", "Awa", correction="شريت")
    registre.valider(i, "confirmee", "Hinimdou")
    p = registre.prediction(i)
    assert p["statut"] == "confirmee"
    assert p["derniere"]["validateur"] == "Hinimdou"
    assert p["derniere"]["correction"] is None
    lignes, total = registre.historique()
    assert total == 2
    assert [x["decision"] for x in lignes] == ["confirmee", "corrigee"]
    assert lignes[1]["correction"] == "شريت" and lignes[1]["entree"] == "chrit"


@pytest.mark.parametrize("decision, options, message", [
    ("validee", {}, "inconnue"),
    ("corrigee", {"correction": "  "}, "correction"),
    ("ecartee", {}, "motif"),
    ("ecartee", {"motif": "autre"}, "motif"),
])
def test_valider_refuse_l_incomplet(registre, decision, options, message):
    [i] = registre.enregistrer([ligne("chrit", 0.3)])
    with pytest.raises(ValueError, match=message):
        registre.valider(i, decision, "Awa", **options)
    assert registre.prediction(i)["statut"] == "a_valider"
    assert registre.historique()[1] == 0


def test_valider_sans_nom_refuse(registre):
    [i] = registre.enregistrer([ligne("chrit", 0.3)])
    with pytest.raises(ValueError, match="validateur"):
        registre.valider(i, "confirmee", "  ")


def test_prediction_inconnue(registre):
    assert registre.prediction(999) is None
    with pytest.raises(LookupError):
        registre.valider(999, "confirmee", "Awa")


def test_vue_registre_erreurs(registre):
    a, b, c = registre.enregistrer([ligne("a", 0.1), ligne("b", 0.2), ligne("c", 0.3)])
    registre.valider(a, "corrigee", "Awa", correction="أ")
    registre.valider(a, "corrigee", "Awa", correction="ا")
    registre.valider(b, "confirmee", "Awa")
    registre.valider(c, "corrigee", "Awa", correction="س")
    registre.valider(c, "ecartee", "Awa", motif="pas_darija")
    with registre.moteur.connect() as conn:
        lignes = conn.execute(text(
            "SELECT prediction_id, correction FROM registre_erreurs")).all()
    assert [tuple(x) for x in lignes] == [(a, "ا")]


def test_a_exporter_suit_la_derniere_decision(registre):
    a, b, c, d = registre.enregistrer(
        [ligne("a", 0.1), ligne("b", 0.2), ligne("c", 0.3), ligne("d", 0.4)])
    registre.valider(a, "corrigee", "Awa", correction="un")
    registre.valider(a, "corrigee", "Awa", correction="deux")
    registre.valider(b, "confirmee", "Awa")
    registre.valider(c, "ecartee", "Awa", motif="donnee_personnelle")
    # d reste a valider : jamais exportee
    assert [(x["id"], x["darija"]) for x in registre.a_exporter()] == [(a, "deux")]
    avec = registre.a_exporter(avec_confirmations=True)
    assert [(x["id"], x["darija"]) for x in avec] == [(a, "deux"), (b, "<b>")]
