"""Normalisation des textes et cles de rapprochement, sans dependance.

Sortie de make_splits.py pour que le registre et l'interface de validation s'en
servent sans charger pandas ni huggingface_hub. make_splits.py les reexporte :
ses appelants n'ont rien a changer.
"""

import re
import unicodedata

# Signes de vocalisation et tatweel : TerjamaBench n'en porte pas (0,2 % des mots),
# alors que doda_synthetic en porte 21 %. Les laisser ferait compter chaque
# diacritique produit comme une erreur de caractère, sans erreur de langue.
VOCALISATION = [chr(c) for c in range(0x064B, 0x0656)] + ["ـ", "ٰ"]
TABLE_VOCAL = {ord(c): None for c in VOCALISATION}
# TerjamaBench écrit le point d'interrogation en arabe dans 100 % des cas,
# DODa-10K en latin dans 97 % des cas.
PONCTUATION_ARABE = {ord("?"): "؟", ord(";"): "؛"}

PONCT_CLE = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "،؛؟٪٫٬٭۔"
    "«»‘’“”…–—"
)
TABLE_PONCT = {ord(c): " " for c in PONCT_CLE}
EQUIVALENCES = [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")]


def normaliser_cible(texte):
    # Ce que le modèle devra produire. Seules les différences non débattues sont
    # corrigées ici ; la convention orthographique proprement dite reste à arbitrer.
    t = unicodedata.normalize("NFKC", str(texte)).translate(TABLE_VOCAL)
    return re.sub(r"\s+", " ", t.translate(PONCTUATION_ARABE)).strip()


def cle_arabe(texte):
    t = unicodedata.normalize("NFKC", str(texte)).translate(TABLE_PONCT).translate(TABLE_VOCAL)
    for avant, apres in EQUIVALENCES:
        t = t.replace(avant, apres)
    return re.sub(r"\s+", " ", t).strip()


def cle_latine(texte):
    t = unicodedata.normalize("NFKC", str(texte)).lower().translate(TABLE_PONCT)
    return re.sub(r"\s+", " ", t).strip()


def contient_vocalisation(texte):
    """Vrai si le texte porte un signe que normaliser_cible retirerait."""
    return any(ord(c) in TABLE_VOCAL for c in str(texte))
