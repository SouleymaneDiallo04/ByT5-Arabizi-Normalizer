"""Normalisation partagee par le decoupage, le registre et l'interface."""

import subprocess
import sys
from pathlib import Path

from donnees import make_splits, normalisation

RACINE = Path(__file__).resolve().parent.parent


def test_make_splits_reexporte_les_memes_fonctions():
    assert make_splits.normaliser_cible is normalisation.normaliser_cible
    assert make_splits.cle_arabe is normalisation.cle_arabe
    assert make_splits.cle_latine is normalisation.cle_latine


def test_normaliser_cible():
    assert normalisation.normaliser_cible("واش  كاين?") == "واش كاين؟"
    assert normalisation.normaliser_cible("شْريت") == "شريت"


def test_cles():
    assert normalisation.cle_arabe("أنا، هنا!") == "انا هنا"
    assert normalisation.cle_latine("Ana, HNA!") == "ana hna"


def test_contient_vocalisation():
    assert normalisation.contient_vocalisation("شْريت")
    assert normalisation.contient_vocalisation("كـبير")
    assert not normalisation.contient_vocalisation("شريت ڭاع، واش؟")


def test_sans_pandas():
    """L'image de validation n'a ni pandas ni huggingface_hub."""
    code = ("import sys, donnees.normalisation; "
            "print('pandas' in sys.modules, 'huggingface_hub' in sys.modules)")
    sortie = subprocess.run([sys.executable, "-c", code], cwd=RACINE,
                            capture_output=True, text=True, check=True)
    assert sortie.stdout.split() == ["False", "False"]
