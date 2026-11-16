"""Tests du service, sans le modele : un faux normaliseur le remplace.

Ce qui est verifie ici est le contrat de l'API (EF-03, EF-04, EF-06) et le
calcul du rang. La justesse du modele se mesure ailleurs, par evaluation/.
Lancer depuis la racine du depot : python -m pytest
"""

import math

import pytest
import torch
from fastapi.testclient import TestClient

from modele.inference import confiance_moyenne
from service.api import creer_app
from service.calibration import Calibration
from service.normaliseur import MAX_OCTETS

TRANSCRIPTION = {"chrit 3 tomobilat": "شريت 3 طوموبيلات", "kanbghik bzaf": "كنبغيك بزاف"}


class FauxNormaliseur:
    nom, revision, appareil = "modele-test", "rev-test", "cpu"

    def __init__(self):
        self.recus = []

    def normaliser(self, textes):
        self.recus.append(list(textes))
        # Confiance differente par phrase, pour que l'ordre se voie
        return [(TRANSCRIPTION.get(t, t.upper()), 0.90 + 0.01 * i, i != 2)
                for i, t in enumerate(textes)]


CALIBRATION = Calibration(tuple(0.90 + 0.001 * i for i in range(100)),
                          "modele-test", "rev-test")


@pytest.fixture
def faux():
    return FauxNormaliseur()


@pytest.fixture
def client(faux):
    with TestClient(creer_app(faux, CALIBRATION)) as c:
        yield c


def test_une_phrase(client):
    r = client.post("/normaliser", json={"texte": "chrit 3 tomobilat"})
    assert r.status_code == 200
    d = r.json()
    assert d["entree"] == "chrit 3 tomobilat"
    assert d["sortie"] == "شريت 3 طوموبيلات"
    assert 0 <= d["confiance"] <= 1 and 0 <= d["confiance_brute"] <= 1
    assert d["confiance"] == CALIBRATION.rang(0.90)
    assert d["complete"] is True


def test_sortie_en_utf8(client):
    """EF-01 : la reponse est de l'UTF-8 valide, sans echappement \\uXXXX."""
    r = client.post("/normaliser", json={"texte": "kanbghik bzaf"})
    assert "كنبغيك".encode("utf-8") in r.content


def test_lot_rend_n_resultats_dans_l_ordre(client):
    textes = ["kanbghik bzaf", "chrit 3 tomobilat", "wach", "daba"]
    r = client.post("/normaliser/lot", json={"textes": textes})
    assert r.status_code == 200
    resultats = r.json()["resultats"]
    assert [x["entree"] for x in resultats] == textes
    assert [x["sortie"] for x in resultats] == [
        "كنبغيك بزاف", "شريت 3 طوموبيلات", "WACH", "DABA"]
    rangs = [x["confiance"] for x in resultats]
    assert rangs == sorted(rangs)
    assert [x["complete"] for x in resultats] == [True, True, False, True]


def test_entree_nettoyee_comme_a_l_entrainement(client, faux):
    r = client.post("/normaliser", json={"texte": "  chrit\t3   tomobilat \n"})
    assert faux.recus[-1] == ["chrit 3 tomobilat"]
    assert r.json()["entree"] == "  chrit\t3   tomobilat \n"


@pytest.mark.parametrize("texte", ["", "   ", "\n\t"])
def test_texte_vide_refuse(client, texte):
    assert client.post("/normaliser", json={"texte": texte}).status_code == 422


def test_limite_en_octets(client):
    assert client.post("/normaliser", json={"texte": "a" * MAX_OCTETS}).status_code == 200
    assert client.post("/normaliser", json={"texte": "a" * (MAX_OCTETS + 1)}).status_code == 422
    # La limite porte sur les octets : un caractere accentue en compte deux
    trop = "é" * (MAX_OCTETS // 2 + 1)
    r = client.post("/normaliser", json={"texte": trop})
    assert r.status_code == 422 and "octets" in r.text


def test_lot_borne(client):
    from service.config import MAX_LOT
    assert client.post("/normaliser/lot", json={"textes": []}).status_code == 422
    assert client.post("/normaliser/lot",
                       json={"textes": ["daba"] * (MAX_LOT + 1)}).status_code == 422
    r = client.post("/normaliser/lot", json={"textes": ["daba", " "]})
    assert r.status_code == 422 and "phrase 1" in r.text


def test_openapi_et_swagger(client):
    """EF-04 : schema OpenAPI publie et page Swagger servie."""
    chemins = client.get("/openapi.json").json()["paths"]
    assert {"/normaliser", "/normaliser/lot", "/sante"} <= set(chemins)
    assert client.get("/docs").status_code == 200


def test_sante(client):
    d = client.get("/sante").json()
    assert d["statut"] == "pret" and d["revision"] == "rev-test"
    assert d["phrases_calibration"] == 100


def test_calibration_d_un_autre_modele_refusee(faux):
    autre = Calibration(CALIBRATION.scores, "modele-test", "autre-revision")
    with pytest.raises(RuntimeError, match="calibration"):
        with TestClient(creer_app(faux, autre)):
            pass


def test_rang():
    c = Calibration((0.1, 0.2, 0.3, 0.4), "m", "r")
    assert c.rang(0.05) == 0.0
    assert c.rang(0.99) == 1.0
    assert c.rang(0.25) == 0.5
    assert c.rang(0.2) == 0.38       # ex aequo compte pour moitie : 1,5 sur 4


def test_calibration_invalide():
    with pytest.raises(ValueError):
        Calibration((), "m", "r")
    with pytest.raises(ValueError):
        Calibration((0.3, 0.2), "m", "r")
    with pytest.raises(ValueError):
        Calibration((0.5, 1.2), "m", "r")


def test_confiance_moyenne_ignore_le_remplissage():
    """Le jeton 0 qui complete une phrase deja finie ne compte pas."""
    jetons = torch.tensor([[5, 6, 1], [5, 1, 0]])
    probas = torch.tensor([[0.5, 0.5, 0.5], [0.8, 0.8, 1.0]])
    c = confiance_moyenne(probas.log(), jetons, pad_id=0)
    assert math.isclose(c[0].item(), 0.5, rel_tol=1e-6)
    assert math.isclose(c[1].item(), 0.8, rel_tol=1e-6)
