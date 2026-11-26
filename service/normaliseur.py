"""Le modele, charge une fois au demarrage et partage par les requetes."""

import re
import threading
import unicodedata

from modele.inference import MAX_SOURCE, decoder
from service.config import MODELE, QUANTIFIE, REVISION

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

    def __init__(self, modele=MODELE, revision=REVISION, appareil=None,
                 quantifie=QUANTIFIE):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        # La quantification dynamique de torch n'existe que sur processeur.
        if quantifie and appareil not in (None, "cpu"):
            raise ValueError(f"modele quantifie demande sur {appareil} : la "
                             "quantification dynamique ne tourne que sur "
                             "processeur (QUANTIFIE=0 pour le GPU)")
        if quantifie:
            appareil = "cpu"
        self.nom, self.revision = modele, revision
        self.appareil = appareil or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(modele, revision=revision)
        self.modele = (AutoModelForSeq2SeqLM.from_pretrained(modele, revision=revision)
                       .to(self.appareil).eval())
        self.variante = "fp32"
        if quantifie:
            import gc

            from modele.quantification import quantifier_modele
            self.modele, _ = quantifier_modele(self.modele)
            # La quantification travaille sur une copie. L'original en pleine
            # precision ne disparait qu'au passage du ramasse-miettes, ses
            # modules se referencant entre eux : sans cet appel, il restait
            # en memoire tant que le service tournait.
            gc.collect()
            self.variante = "int8"
        # Une requete a la fois. Sur processeur, une seule phrase occupe deja
        # toute la machine : en decoder deux ensemble les ralentirait toutes
        # les deux, sans rien gagner.
        self._verrou = threading.Lock()

    def normaliser(self, textes):
        """Pour chaque texte, dans l'ordre : (sortie, confiance brute, complete).

        En int8, les phrases passent une a une. La quantification dynamique
        calcule l'echelle des activations sur tout le lot, remplissage compris :
        decodee avec d'autres, une phrase change de sortie dans 11 cas sur 64
        (journal du 26 novembre). La meme entree doit rendre la meme sortie,
        seule ou en lot. Le fp32 n'a pas ce defaut et garde ses lots.
        """
        with self._verrou:
            sorties, confiances, completes = decoder(
                self.modele, self.tok, textes, self.appareil,
                **({"taille_lot": 1} if self.variante == "int8" else {}))
        return list(zip(sorties, confiances, completes))
