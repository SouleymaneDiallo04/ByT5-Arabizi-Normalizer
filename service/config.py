"""Reglages du service, modifiables par variables d'environnement."""

import os

MODELE = os.environ.get("MODELE", "sawti-nlp/byt5-arabizi-normalizer")

# Revision figee du depot Hugging Face : celle du melange 3 et de sa fiche
# (ENF-04). La calibration est liee a cette revision, et le service refuse de
# demarrer si elles ne concordent pas. Pour un modele copie en local, MODELE est
# un chemin et MODELE_REVISION doit dire de quelle revision il provient.
REVISION = os.environ.get("MODELE_REVISION",
                          "96a58601466f5aa0e72ba84ab418d8e4c0394e27")

CALIBRATION = os.environ.get(
    "CALIBRATION", os.path.join(os.path.dirname(__file__), "calibration.json"))

# Phrases par requete de lot (EF-06). Au-dela, mieux vaut plusieurs requetes :
# sur processeur, un lot de 16 prend deja une quinzaine de secondes.
MAX_LOT = int(os.environ.get("MAX_LOT", 64))
