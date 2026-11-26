"""Tests de l'interface de validation, sur SQLite, sans modele."""

import re

import pytest
from fastapi.testclient import TestClient

from registre.depot import Registre
from validation.app import creer_app

SORTIE = "شريت 3 طوموبيلات"


def ligne(entree, sortie, confiance):
    return {"entree": entree, "entree_propre": entree, "sortie": sortie,
            "confiance": confiance, "confiance_brute": 0.95, "complete": True,
            "revision": "rev-test"}


@pytest.fixture
def registre(tmp_path):
    r = Registre(f"sqlite:///{(tmp_path / 'registre.db').as_posix()}")
    r.creer_tables()
    return r


@pytest.fixture
def client(registre):
    with TestClient(creer_app(registre)) as c:
        yield c


@pytest.fixture
def ids(registre):
    return registre.enregistrer([
        ligne("chrit 3 tomobilat", SORTIE, 0.80),
        ligne("kanbghik bzaf", "كنبغيك بزاف", 0.10),
        ligne("wach nta mzyan", "واش نتا مزيان", 0.50)])


def decider(client, i, **champs):
    donnees = {"validateur": "Awa", "texte": "", "motif": "", **champs}
    return client.post(f"/prediction/{i}", data=donnees, follow_redirects=False)


def erreur(page):
    m = re.search(r'<p class="erreur" role="alert">(.*?)</p>', page, re.S)
    return m.group(1) if m else None


def test_liste_triee_par_confiance_croissante(client, ids):
    page = client.get("/").text
    positions = [page.index(x) for x in
                 ("kanbghik bzaf", "wach nta mzyan", "chrit 3 tomobilat")]
    assert positions == sorted(positions)
    assert "3 prédictions à valider" in page


def test_liste_par_pages_de_vingt(client, registre):
    registre.enregistrer([ligne(f"jomla {i}", "جملة", i / 100) for i in range(25)])
    assert client.get("/").text.count('class="prediction"') == 20
    assert client.get("/?page=2").text.count('class="prediction"') == 5


def test_suivante_ouvre_la_moins_sure(client, ids):
    r = client.get("/suivante", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/prediction/{ids[1]}"


def test_suivante_sans_rien_a_valider(client):
    assert client.get("/suivante", follow_redirects=False).headers["location"] == "/"


def test_page_pre_remplie_de_droite_a_gauche(client, ids):
    page = client.get(f"/prediction/{ids[0]}").text
    assert re.search(rf'<textarea[^>]*dir="rtl"[^>]*>{SORTIE}</textarea>', page)
    for lettre in "ڭڤ؟،":
        assert f'data-lettre="{lettre}"' in page


def test_confirmer(client, registre, ids):
    r = decider(client, ids[0], decision="confirmee", texte=SORTIE)
    assert r.status_code == 303 and r.headers["location"] == "/suivante"
    assert registre.prediction(ids[0])["statut"] == "confirmee"


def test_correction_visible_apres_rechargement(client, registre, ids):
    decider(client, ids[0], decision="corrigee", texte="شريت  3 طوموبيلات ديال\r\n")
    p = registre.prediction(ids[0])
    assert p["statut"] == "corrigee"
    assert p["derniere"]["correction"] == "شريت 3 طوموبيلات ديال"
    assert ">شريت 3 طوموبيلات ديال</textarea>" in client.get(f"/prediction/{ids[0]}").text


def test_ecarter_avec_motif(client, registre, ids):
    decider(client, ids[2], decision="ecartee", motif="donnee_personnelle")
    p = registre.prediction(ids[2])
    assert p["statut"] == "ecartee" and p["derniere"]["motif"] == "donnee_personnelle"


@pytest.mark.parametrize("champs, message", [
    ({"decision": "ecartee"}, "motif"),
    ({"decision": "confirmee", "texte": "شي حاجة اخرى"}, "Corriger"),
    ({"decision": "corrigee", "texte": SORTIE}, "Confirmer"),
    ({"decision": "corrigee", "texte": "   "}, "Écarter"),
    ({"decision": "corrigee", "texte": "شْريت 3 طوموبيلات"}, "vocalisation"),
    ({"decision": "confirmee", "texte": SORTIE, "validateur": " "}, "nom"),
    ({"decision": "validee", "texte": SORTIE}, "inconnue"),
])
def test_decisions_refusees(client, registre, ids, champs, message):
    r = decider(client, ids[0], **champs)
    assert r.status_code == 422 and message in erreur(r.text)
    assert registre.prediction(ids[0])["statut"] == "a_valider"
    assert registre.historique()[1] == 0


def test_saisie_conservee_apres_refus(client, ids):
    page = decider(client, ids[0], decision="corrigee", texte="شْريت").text
    assert ">شْريت</textarea>" in page


def test_nom_garde_dans_un_cookie(client, ids):
    decider(client, ids[0], decision="confirmee", texte=SORTIE, validateur="سعاد")
    assert 'value="سعاد"' in client.get(f"/prediction/{ids[1]}").text


def test_historique_garde_chaque_decision(client, registre, ids):
    decider(client, ids[0], decision="corrigee", texte="شريت 3 طوموبيلات ديال")
    decider(client, ids[0], decision="confirmee", texte=SORTIE, depuis="historique")
    page = client.get("/historique").text
    assert page.count('class="decision"') == 2
    assert f"/prediction/{ids[0]}?depuis=historique" in page
    assert registre.prediction(ids[0])["statut"] == "confirmee"


def test_retour_vers_l_historique(client, ids):
    r = decider(client, ids[0], decision="confirmee", texte=SORTIE, depuis="historique")
    assert r.headers["location"] == "/historique"


def test_retour_inconnu_ignore(client, ids):
    r = decider(client, ids[0], decision="confirmee", texte=SORTIE,
                depuis="https://exemple.org")
    assert r.headers["location"] == "/suivante"


def test_prediction_inconnue(client):
    assert client.get("/prediction/999").status_code == 404
    assert decider(client, 999, decision="confirmee").status_code == 404


def test_texte_echappe(client, registre):
    [i] = registre.enregistrer([ligne("<script>alert(1)</script>", "x", 0.1)])
    page = client.get(f"/prediction/{i}").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_sans_database_url_refuse_de_demarrer(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        with TestClient(creer_app()):
            pass
