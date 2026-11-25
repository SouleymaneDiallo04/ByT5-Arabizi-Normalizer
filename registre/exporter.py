"""Export du registre pour le reentrainement.

Usage, la base designee par DATABASE_URL :
    python -m registre.exporter registre.csv
    python -m registre.exporter registre.csv --avec-confirmations --test test_real.csv

transliteration et darija_norm sont les colonnes que lit modele/train_byt5.py ;
id, darija et source celles des autres corpus, pour que le fichier se melange
comme eux. sortie_modele, confiance et revision servent a l'analyse.

ATTENTION. Toute phrase du jeu de test est retiree, avec les cles de
make_splits.py. Sans ce garde-fou, une phrase de TerjamaBench soumise au
service puis corrigee entrerait dans l'entrainement, et l'evaluation ne
vaudrait plus rien.
"""

import argparse
import csv
import os

from donnees.normalisation import cle_arabe, cle_latine, normaliser_cible
from registre.depot import Registre

COLONNES = ["id", "transliteration", "darija_norm", "darija", "source",
            "sortie_modele", "confiance", "revision"]
TEST = "test_real.csv"


def cles_du_test(chemin):
    try:
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            lignes = list(csv.DictReader(f))
    except FileNotFoundError:
        raise SystemExit(
            f"{chemin} absent : sans lui, rien ne garantit que l'export "
            "epargne le jeu de test. Le produire par python -m "
            "donnees.make_splits, ou le telecharger :\n"
            "  hf download sawti-nlp/darija-arabizi-corpus splits/test_real.csv "
            "--repo-type dataset --local-dir .\n"
            "puis passer --test splits/test_real.csv") from None
    return ({cle_arabe(x["darija"]) for x in lignes},
            {cle_latine(x["transliteration"]) for x in lignes})


def exporter(registre, sortie, test=TEST, avec_confirmations=False):
    """Ecrit le fichier d'entrainement et rend le bilan des lignes."""
    cles_ar, cles_la = cles_du_test(test)
    lignes = registre.a_exporter(avec_confirmations)
    gardees = [x for x in lignes
               if cle_arabe(x["darija"]) not in cles_ar
               and cle_latine(x["entree_propre"]) not in cles_la]
    with open(sortie, "w", encoding="utf-8-sig", newline="") as f:
        ecrivain = csv.DictWriter(f, fieldnames=COLONNES)
        ecrivain.writeheader()
        for x in gardees:
            ecrivain.writerow({
                "id": f"registre_{x['id']:06d}",
                "transliteration": x["entree_propre"],
                "darija_norm": normaliser_cible(x["darija"]),
                "darija": x["darija"],
                "source": "registre",
                "sortie_modele": x["sortie"],
                "confiance": x["confiance"],
                "revision": x["revision"],
            })
    return {"corrigees": sum(x["statut"] == "corrigee" for x in lignes),
            "confirmees": sum(x["statut"] == "confirmee" for x in lignes),
            "retirees_test": len(lignes) - len(gardees),
            "ecrites": len(gardees)}


def main(arguments=None):
    p = argparse.ArgumentParser(
        description="Exporte le registre de validation pour le réentraînement.")
    p.add_argument("sortie", help="fichier CSV à écrire")
    p.add_argument("--avec-confirmations", action="store_true",
                   help="ajouter les sorties confirmées, comme exemples justes")
    p.add_argument("--test", default=TEST,
                   help="jeu de test à épargner (défaut : test_real.csv)")
    a = p.parse_args(arguments)
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL non définie : quelle base exporter ?")
    bilan = exporter(Registre(url), a.sortie, a.test, a.avec_confirmations)
    print(a.sortie)
    print(f"  corrigées             {bilan['corrigees']:>6}")
    print(f"  confirmées            {bilan['confirmees']:>6}")
    print(f"  retirées, jeu de test {bilan['retirees_test']:>6}")
    print(f"  écrites               {bilan['ecrites']:>6}")
    return bilan


if __name__ == "__main__":
    main()
