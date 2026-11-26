"""Regles de service/pretraitement.py, avec un faux modele."""

import pytest

from service.pretraitement import (appliquer, decouper, reduire_repetitions,
                                   retirer_emojis)


def faux(textes):
    """Rend chaque texte entre chevrons ; confiance selon la longueur."""
    return [(f"<{t}>", 1 / (1 + len(t)), not t.endswith("!")) for t in textes]


def test_reduire_repetitions():
    assert reduire_repetitions("bzaaaaf") == "bzaaf"
    assert reduire_repetitions("AAAAch kifek") == "AAch kifek"
    # Le doublement a un sens, les chiffres ne sont pas des lettres.
    assert reduire_repetitions("mzyyan bzzaf 6000") == "mzyyan bzzaf 6000"


def test_retirer_emojis():
    assert retirer_emojis("safi 😂 wakha ❤️") == ("safi wakha", "😂❤️")
    assert retirer_emojis("safi") == ("safi", "")


def test_decouper():
    assert decouper("salam. labas? hamdulah") == ["salam.", "labas?", "hamdulah"]
    # Un point sans espace apres n'est pas une fin de phrase.
    assert decouper("jabt 18.5 f examen") == ["jabt 18.5 f examen"]


def test_sans_regle_rien_ne_change():
    assert appliquer(faux, ["a. b"], ()) == faux(["a. b"])


def test_phrases_rejointes_confiance_la_plus_basse():
    (sortie, confiance, complete), = appliquer(faux, ["ab. c!"], ("phrases",))
    assert sortie == "<ab.> <c!>"
    assert confiance == pytest.approx(min(1 / 4, 1 / 3))
    assert complete is False


def test_emojis_rendus_a_la_fin():
    assert appliquer(faux, ["safi 😂 wakha"], ("emojis",))[0][0] == "<safi wakha> 😂"
    # Un message fait seulement d'emojis ne passe pas par le modele.
    assert appliquer(faux, ["🙏"], ("emojis",)) == [("🙏", 1.0, True)]


def test_regles_combinees_et_ordre_garde():
    r = appliquer(faux, ["bzaaaaf. ok 👍", "wakha"], ("repetitions", "emojis", "phrases"))
    assert [s for s, _, _ in r] == ["<bzaaf.> <ok> 👍", "<wakha>"]


def test_regle_inconnue():
    with pytest.raises(ValueError):
        appliquer(faux, ["a"], ("minuscules",))
