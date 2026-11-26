"""Export du registre : colonnes, cible normalisee, garde-fou du jeu de test."""

import csv

import pytest

from registre.depot import Registre
from registre.exporter import COLONNES, exporter, main


def ligne(entree, sortie):
    return {"entree": f"  {entree} ", "entree_propre": entree, "sortie": sortie,
            "confiance": 0.42, "confiance_brute": 0.95, "complete": True,
            "revision": "rev-test"}


@pytest.fixture
def registre(tmp_path):
    r = Registre(f"sqlite:///{(tmp_path / 'registre.db').as_posix()}")
    r.creer_tables()
    return r


@pytest.fixture
def jeu_de_test(tmp_path):
    chemin = tmp_path / "test_real.csv"
    with open(chemin, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "darija", "transliteration"])
        w.writerow(["terjama_0000", "الله يسعد الماسا", "lays3d lmasa"])
    return chemin


def lire(chemin):
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_colonnes_et_cible_normalisee(registre, jeu_de_test, tmp_path):
    [i] = registre.enregistrer([ligne("wach kayn", "واش كاين")])
    registre.valider(i, "corrigee", "Awa", correction="واش كاين?")
    sortie = tmp_path / "registre.csv"
    bilan = exporter(registre, sortie, jeu_de_test)
    [x] = lire(sortie)
    assert list(x) == COLONNES
    assert x["transliteration"] == "wach kayn"      # l'entree vue par le modele
    assert x["darija"] == "واش كاين?"
    assert x["darija_norm"] == "واش كاين؟"
    assert x["sortie_modele"] == "واش كاين"
    assert x["id"] == f"registre_{i:06d}" and x["source"] == "registre"
    assert x["revision"] == "rev-test" and float(x["confiance"]) == 0.42
    assert bilan == {"corrigees": 1, "confirmees": 0, "retirees_test": 0,
                     "ecrites": 1}


def test_ecartees_exclues_confirmations_sur_demande(registre, jeu_de_test, tmp_path):
    a, b, c, _ = registre.enregistrer([ligne("a1", "ا1"), ligne("b1", "ب1"),
                                       ligne("c1", "ج1"), ligne("d1", "د1")])
    registre.valider(a, "corrigee", "Awa", correction="ا2")
    registre.valider(b, "confirmee", "Awa")
    registre.valider(c, "ecartee", "Awa", motif="pas_darija")
    sortie = tmp_path / "registre.csv"
    exporter(registre, sortie, jeu_de_test)
    assert [x["transliteration"] for x in lire(sortie)] == ["a1"]
    bilan = exporter(registre, sortie, jeu_de_test, avec_confirmations=True)
    lignes = lire(sortie)
    assert [(x["transliteration"], x["darija"]) for x in lignes] == [
        ("a1", "ا2"), ("b1", "ب1")]
    assert bilan["confirmees"] == 1 and bilan["ecrites"] == 2


def test_phrases_du_test_retirees(registre, jeu_de_test, tmp_path):
    a, b, c = registre.enregistrer([
        ligne("Lays3d lmasa!", "اللاه يسعد"),      # meme arabizi, casse et ponctuation a part
        ligne("autre chose", "الله يسعد"),          # meme arabe, une fois corrige
        ligne("chrit", "شرط")])
    registre.valider(a, "corrigee", "Awa", correction="الله يسعد المسا")
    registre.valider(b, "corrigee", "Awa", correction="الله يسعد الماسا.")
    registre.valider(c, "corrigee", "Awa", correction="شريت")
    sortie = tmp_path / "registre.csv"
    bilan = exporter(registre, sortie, jeu_de_test)
    assert [x["transliteration"] for x in lire(sortie)] == ["chrit"]
    assert bilan["retirees_test"] == 2 and bilan["ecrites"] == 1


def test_jeu_de_test_absent(registre, tmp_path):
    with pytest.raises(SystemExit, match="make_splits"):
        exporter(registre, tmp_path / "x.csv", tmp_path / "absent.csv")


def test_ligne_de_commande(registre, jeu_de_test, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", str(registre.moteur.url))
    [i] = registre.enregistrer([ligne("chrit", "شريت")])
    registre.valider(i, "confirmee", "Awa")
    bilan = main([str(tmp_path / "r.csv"), "--test", str(jeu_de_test),
                  "--avec-confirmations"])
    assert bilan["ecrites"] == 1
    assert "écrites" in capsys.readouterr().out


def test_ligne_de_commande_sans_base(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit, match="DATABASE_URL"):
        main([str(tmp_path / "r.csv")])
