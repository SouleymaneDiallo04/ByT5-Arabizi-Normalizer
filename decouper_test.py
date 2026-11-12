"""Prelevement d'un jeu de reglage dans TerjamaBench.

Pourquoi. Le jeu de developpement issu de \x60make_splits.py\x60 vient des memes
sources que l'entrainement, et le modele y reussit deux fois et demie mieux que
sur le jeu de test : 0,0473 de CER contre 0,1146. Regler quoi que ce soit
dessus revient a optimiser sur une distribution ou le probleme n'existe pas.
C'est ce qui a fait echouer le reordonnanceur, dont les poids, appris sur des
faisceaux faciles, ne transferaient pas.

\x60dev.csv\x60 garde son role d'arret precoce pendant l'entrainement, ou il mesure
la convergence et non la qualite finale. Mais toute decision de selection
jugee au CER doit se regler sur une distribution de test.

Ce qu'il coute. Les tranches regionales sont deja minces : 21 phrases pour
Nador, 25 pour Guelmim. A 20 %, elles tombent a 17 et 20, ce qui reste lisible.
A 50 % elles tomberaient a 11 et 13, et l'analyse par variete regionale, qui a
produit le principal resultat du projet, deviendrait du bruit. D'ou ce taux et
pas un autre.

Le prelevement est stratifie sur le couple theme et variete regionale, afin que
les deux axes d'analyse gardent leurs proportions. Il est seme, donc rejouable,
comme l'exige ENF-04.

\x60test_evaluation.csv\x60 ne doit servir qu'a mesurer, jamais a choisir. Les
chiffres publies avant ce decoupage portaient sur les 850 phrases et ne sont
pas comparables a ceux qui porteront sur 680 : ce script les recalcule donc sur
la nouvelle base pour tout fichier de predictions trouve a cote de lui.

Usage :
    python decouper_test.py
"""

import glob
import os
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

from evaluate import normaliser, mesurer

DEPOT = "sawti-nlp/darija-arabizi-corpus"
TEST = "test_real.csv"
REGLAGE = "test_reglage.csv"
EVALUATION = "test_evaluation.csv"

PART_REGLAGE = 0.20
GRAINE = 42
STRATES = ["topic", "annotator_dialect"]


def charger(chemin):
    try:
        return pd.read_csv(chemin)
    except FileNotFoundError:
        return pd.read_csv(hf_hub_download(DEPOT, f"splits/{chemin}",
                                           repo_type="dataset"))


def prelever(test, part=PART_REGLAGE):
    """Tire un sous-ensemble stratifie sur le couple theme et variete.

    L'arrondi est fait au plus proche et non au superieur : arrondir vers le
    haut sur une centaine de strates minuscules gonflerait le prelevement bien
    au-dela du taux demande.
    """
    retenus = []
    for _, groupe in test.groupby(STRATES, sort=False):
        combien = round(len(groupe) * part)
        if combien:
            retenus.append(groupe.sample(combien, random_state=GRAINE))
    reglage = pd.concat(retenus).sort_index()
    return reglage, test.drop(reglage.index)


def comparer(test, reglage, evaluation, colonne):
    """Verifie que le decoupage n'a pas deforme une des deux ventilations."""
    print(f"\n{colonne}")
    print(f"{'':<22} {'depart':>7} {'reglage':>9} {'evaluation':>11} {'part':>7}")
    for valeur, n in test[colonne].value_counts().items():
        r = int((reglage[colonne] == valeur).sum())
        e = int((evaluation[colonne] == valeur).sum())
        print(f"{str(valeur)[:22]:<22} {n:>7} {r:>9} {e:>11} {100 * r / n:>6.1f}%")


if __name__ == "__main__":
    test = charger(TEST)
    reglage, evaluation = prelever(test)

    print(f"{TEST} : {len(test)} phrases")
    print(f"  {REGLAGE:<22} {len(reglage):>4} phrases "
          f"({100 * len(reglage) / len(test):.1f} %)")
    print(f"  {EVALUATION:<22} {len(evaluation):>4} phrases "
          f"({100 * len(evaluation) / len(test):.1f} %)")
    assert set(reglage["id"]) & set(evaluation["id"]) == set(), \
        "les deux moities se recoupent"
    assert len(reglage) + len(evaluation) == len(test), "des phrases ont disparu"

    for colonne in STRATES:
        comparer(test, reglage, evaluation, colonne)

    for nom, part in [(REGLAGE, reglage), (EVALUATION, evaluation)]:
        part.to_csv(nom, index=False, encoding="utf-8-sig")

    # Les chiffres publies jusqu'ici portaient sur 850 phrases. Tout ce qui
    # suivra portera sur 680, et les deux series ne sont pas comparables.
    fichiers = sorted(f for f in glob.glob("*.csv")
                      if "prediction" in f or f.startswith("byt5_"))
    if not fichiers:
        sys.exit(0)
    garder = set(evaluation["id"])
    print(f"\nChiffres de reference recalcules sur les {len(evaluation)} "
          f"phrases d'evaluation")
    print(f"{'fichier':<30} {'CER 850':>9} {'CER 680':>9} {'exact. 850':>11} "
          f"{'exact. 680':>11}")
    for fichier in fichiers:
        d = pd.read_csv(fichier)
        if "prediction" not in d.columns or "id" not in d.columns:
            continue
        reduit = d[d["id"].isin(garder)]
        if len(reduit) != len(evaluation):
            continue
        tout = mesurer(d["darija_norm"].map(normaliser),
                       d["prediction"].astype(str).map(normaliser))
        part = mesurer(reduit["darija_norm"].map(normaliser),
                       reduit["prediction"].astype(str).map(normaliser))
        print(f"{os.path.basename(fichier)[:30]:<30} {tout['cer']:>9.4f} "
              f"{part['cer']:>9.4f} {100 * tout['exactitude']:>10.1f}% "
              f"{100 * part['exactitude']:>10.1f}%")
