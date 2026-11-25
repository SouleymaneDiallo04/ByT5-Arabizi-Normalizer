"""Calibre le score de confiance du modele servi (PO-05).

Decode un echantillon du jeu de developpement avec le modele et la revision du
service, et ecrit leurs confiances triees dans service/calibration.json. Le
service s'en sert pour convertir chaque confiance en rang.

A relancer a chaque changement de modele : le fichier porte la revision et la
variante, fp32 ou int8, qu'il calibre, et le service refuse de demarrer sur
une autre.

Le modele servi etant quantifie, la calibration tourne sur processeur, avec le
meme torch que le service : dans un conteneur jetable tire de son image (voir
le README, section Service).

Usage, depuis la racine du depot :
    python -m service.calibrer
"""

import json

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from evaluation.evaluate import normaliser
from evaluation.mesures import DEPOT_DONNEES, ECHANTILLON_DEV, GRAINE, table_confiance
from service.config import CALIBRATION
from service.normaliseur import Normaliseur

FICHIER = "splits/dev.csv"


def main():
    # Revision du jeu de donnees figee et notee : reconstruire le corpus change
    # dev.csv, et la calibration doit pouvoir dire sur quoi elle a ete faite.
    revision_donnees = HfApi().dataset_info(DEPOT_DONNEES).sha
    dev = pd.read_csv(hf_hub_download(DEPOT_DONNEES, FICHIER, repo_type="dataset",
                                      revision=revision_donnees))
    # Meme tirage que evaluation/mesures.py, pour que les deux se recoupent.
    dev = dev.sample(ECHANTILLON_DEV, random_state=GRAINE).reset_index(drop=True)

    n = Normaliseur()
    print(f"modele : {n.nom}@{n.revision} en {n.variante} sur {n.appareil}")
    print(f"donnees : {DEPOT_DONNEES}@{revision_donnees}, {len(dev)} phrases")

    resultats = n.normaliser(dev["transliteration"].astype(str).tolist())
    sorties = [s for s, _, _ in resultats]
    confiances = [b for _, b, _ in resultats]
    incompletes = sum(not c for _, _, c in resultats)

    # La table par decile verifie, au passage, que la confiance reste monotone
    # pour ce modele : sans cela, en donner le rang n'aurait aucun sens.
    table_confiance(dev["darija_norm"].map(normaliser),
                    [normaliser(s) for s in sorties], confiances, "developpement")
    print(f"\n  sorties incompletes : {incompletes} sur {len(sorties)}")

    scores = sorted(confiances)
    with open(CALIBRATION, "w", encoding="utf-8") as f:
        json.dump({"modele": n.nom, "revision": n.revision, "variante": n.variante,
                   "donnees": DEPOT_DONNEES, "revision_donnees": revision_donnees,
                   "fichier": FICHIER, "graine": GRAINE, "phrases": len(scores),
                   "scores": scores}, f, indent=1)
    print(f"  {len(scores)} confiances, de {scores[0]:.4f} a {scores[-1]:.4f}, "
          f"ecrites dans {CALIBRATION}")


if __name__ == "__main__":
    main()
