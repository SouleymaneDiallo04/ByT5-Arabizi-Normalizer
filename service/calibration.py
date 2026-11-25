"""Rang calibre du score de confiance (PO-05).

La probabilite moyenne par octet se concentre entre 0,92 et 1 : lue telle
quelle, une confiance de 0,96 ne dit pas qu'elle tombe dans le pire decile. Le
service renvoie donc son rang parmi les confiances du jeu de developpement,
calculees une fois pour toutes par `python -m service.calibrer`.
"""

import bisect
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    """Confiances de reference, triees, et le modele qui les a produites.

    La variante, fp32 ou int8, en fait partie : la quantification deplace les
    confiances, et un rang calcule sur l'une serait faux pour l'autre.
    """

    scores: tuple
    modele: str
    revision: str
    variante: str = "fp32"

    def __post_init__(self):
        if not self.scores:
            raise ValueError("calibration vide")
        if any(not 0.0 <= s <= 1.0 for s in self.scores):
            raise ValueError("calibration hors de [0,1]")
        if list(self.scores) != sorted(self.scores):
            raise ValueError("calibration non triee")

    @classmethod
    def charger(cls, chemin):
        """Lit le fichier ecrit par `service.calibrer`."""
        with open(chemin, encoding="utf-8") as f:
            d = json.load(f)
        # Les fichiers anterieurs a la quantification ne portent pas de
        # variante : ils ont ete calcules en fp32.
        return cls(tuple(d["scores"]), d["modele"], d["revision"],
                   d.get("variante", "fp32"))

    def rang(self, confiance):
        """Part des phrases de developpement moins sures, en centiemes.

        Les ex aequo comptent pour moitie, si bien qu'une confiance egale a la
        mediane rend 0,5. Le resultat est dans [0,1], comme l'exige EF-03.
        """
        dessous = bisect.bisect_left(self.scores, confiance)
        egaux = bisect.bisect_right(self.scores, confiance) - dessous
        return round((dessous + egaux / 2) / len(self.scores), 2)
