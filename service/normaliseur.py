"""Le modele, charge une fois au demarrage et partage par les requetes."""

import re
import threading
import unicodedata

from modele.inference import MAX_SOURCE, decoder
from service.config import MODELE, REVISION

# Le tokenizer ajoute un octet de fin de sequence, compte dans MAX_SOURCE.
MAX_OCTETS = MAX_SOURCE - 1


def nettoyer(texte):
    """Meme nettoyage que les entrees d'entrainement (donnees/merge_all.py) :
    forme NFKC et espaces ramenes a un seul."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", texte)).strip()


class Normaliseur:
    """ByT5 affine, en decodage glouton, avec sa confiance par phrase.

    Les imports lourds sont faits ici et non en tete de module : les tests de
    l'API remplacent cette classe et n'ont pas besoin de transformers.
    """

    def __init__(self, modele=MODELE, revision=REVISION, appareil=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.nom, self.revision = modele, revision
        self.appareil = appareil or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(modele, revision=revision)
        self.modele = (AutoModelForSeq2SeqLM.from_pretrained(modele, revision=revision)
                       .to(self.appareil).eval())
        # Une requete a la fois. Sur processeur, une seule phrase occupe deja
        # toute la machine : en decoder deux ensemble les ralentirait toutes
        # les deux, sans rien gagner.
        self._verrou = threading.Lock()

    def normaliser(self, textes):
        """Pour chaque texte, dans l'ordre : (sortie, confiance brute, complete)."""
        with self._verrou:
            sorties, confiances, completes = decoder(
                self.modele, self.tok, textes, self.appareil)
        return list(zip(sorties, confiances, completes))
