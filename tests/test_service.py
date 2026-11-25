"""Tests du service, sans le modele : un faux normaliseur le remplace.

Ce qui est verifie ici est le contrat de l'API (EF-03, EF-04, EF-06) et le
calcul du rang. La justesse du modele se mesure ailleurs, par evaluation/.
Lancer depuis la racine du depot : python -m pytest
"""

import math

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

import service.api
from modele.inference import confiance_moyenne
from registre.depot import Registre
from service.api import creer_app
from service.calibration import Calibration
from service.normaliseur import MAX_OCTETS

TRANSCRIPTION = {"chrit 3 tomobilat": "شريت 3 طوموبيلات", "kanbghik bzaf": "كنبغيك بزاف"}


class FauxNormaliseur:
    nom, revision, appareil, variante = "modele-test", "rev-test", "cpu", "fp32"

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
    assert d["variante"] == "fp32"


def test_calibration_d_un_autre_modele_refusee(faux):
    autre = Calibration(CALIBRATION.scores, "modele-test", "autre-revision")
    with pytest.raises(RuntimeError, match="calibration"):
        with TestClient(creer_app(faux, autre)):
            pass


def test_calibration_d_une_autre_variante_refusee(faux):
    """La confiance du modele quantifie n'est pas celle du fp32 : un rang
    calcule sur l'une serait faux pour l'autre."""
    quantifiee = Calibration(CALIBRATION.scores, "modele-test", "rev-test", "int8")
    with pytest.raises(RuntimeError, match="int8"):
        with TestClient(creer_app(faux, quantifiee)):
            pass


def test_calibration_lit_sa_variante(tmp_path):
    ancienne = tmp_path / "ancienne.json"
    ancienne.write_text('{"modele": "m", "revision": "r", "scores": [0.5]}')
    assert Calibration.charger(ancienne).variante == "fp32"
    nouvelle = tmp_path / "nouvelle.json"
    nouvelle.write_text('{"modele": "m", "revision": "r", "variante": "int8", '
                        '"scores": [0.5]}')
    assert Calibration.charger(nouvelle).variante == "int8"


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


# --- Registre de validation humaine -------------------------------------------

def url_sqlite(tmp_path):
    return f"sqlite:///{(tmp_path / 'registre.db').as_posix()}"


@pytest.fixture
def registre(tmp_path):
    return Registre(url_sqlite(tmp_path))


@pytest.fixture
def client_registre(faux, registre):
    with TestClient(creer_app(faux, CALIBRATION, registre)) as c:
        yield c


def test_une_prediction_enregistree(client_registre, registre):
    d = client_registre.post(
        "/normaliser", json={"texte": "  chrit 3   tomobilat "}).json()
    p = registre.prediction(d["id"])
    assert p["entree"] == "  chrit 3   tomobilat "
    assert p["entree_propre"] == "chrit 3 tomobilat"
    assert p["sortie"] == d["sortie"] == "شريت 3 طوموبيلات"
    assert p["confiance"] == d["confiance"]
    assert p["confiance_brute"] == d["confiance_brute"]
    assert p["complete"] is True and p["revision"] == "rev-test:fp32"
    assert p["statut"] == "a_valider"


def test_un_lot_enregistre_n_predictions(client_registre, registre):
    textes = ["kanbghik bzaf", "chrit 3 tomobilat", "wach"]
    resultats = client_registre.post(
        "/normaliser/lot", json={"textes": textes}).json()["resultats"]
    ids = [x["id"] for x in resultats]
    assert len(set(ids)) == 3
    assert [registre.prediction(i)["entree"] for i in ids] == textes
    assert [registre.prediction(i)["sortie"] for i in ids] == [
        x["sortie"] for x in resultats]
    assert registre.a_valider()[1] == 3


class RegistreEnPanne(Registre):
    def enregistrer(self, lignes):
        raise OperationalError("INSERT INTO predictions", {},
                               ConnectionRefusedError("base arrêtée"))


def test_base_indisponible_503(faux, tmp_path):
    with TestClient(creer_app(faux, CALIBRATION,
                              RegistreEnPanne(url_sqlite(tmp_path)))) as c:
        r = c.post("/normaliser", json={"texte": "kanbghik bzaf"})
        assert r.status_code == 503 and "registre" in r.json()["detail"]
        lot = c.post("/normaliser/lot", json={"textes": ["wach"]})
        assert lot.status_code == 503


def test_sans_database_url_rien_n_est_enregistre(faux, monkeypatch):
    monkeypatch.setattr(service.api, "DATABASE_URL", None)
    with TestClient(creer_app(faux, CALIBRATION)) as c:
        assert c.post("/normaliser", json={"texte": "wach"}).json()["id"] is None
        assert c.get("/sante").json()["registre"] is False


def test_database_url_active_le_registre(faux, tmp_path, monkeypatch):
    monkeypatch.setattr(service.api, "DATABASE_URL", url_sqlite(tmp_path))
    with TestClient(creer_app(faux, CALIBRATION)) as c:
        d = c.post("/normaliser", json={"texte": "kanbghik bzaf"}).json()
        assert c.get("/sante").json()["registre"] is True
    assert Registre(url_sqlite(tmp_path)).prediction(d["id"])["sortie"] == d["sortie"]


def test_503_documente_dans_openapi(client):
    chemins = client.get("/openapi.json").json()["paths"]
    for route in ("/normaliser", "/normaliser/lot"):
        assert "503" in chemins[route]["post"]["responses"]
