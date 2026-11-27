"""Copies bruitees de l'entrainement : fautes de frappe, voyelles omises, mots colles.

La mesure de robustesse du 26 novembre a montre que le modele tient les
conventions d'ecriture (sh, x, u, q), mais pas les accidents de frappe : une
faute degrade 70 % des phrases, des voyelles omises ou deux mots colles la
moitie. Aucune regle ne les corrige avant le modele. Ce script en montre au
modele pendant l'entrainement : une part des lignes du train est recopiee avec
un arabizi bruite, et la meme cible arabe.

Ce n'est pas l'augmentation du 12 novembre. Celle-ci permutait des graphies
attestees, que le modele tenait deja ; celle-ci imite des accidents.

Les bruits sont ceux de evaluation/robustesse_modele.py, pour que le test
mesure exactement ce qui est appris. Le tirage, lui, est different : la graine
prefixe « entrainement », et les phrases du test ne sont de toute facon jamais
dans le train. Une copie dont l'arabizi bruite tomberait sur une entree du test
est ecartee.

Usage, depuis le dossier des donnees :
    python -m donnees.bruiter_entrainement train_v3.csv train_v3_bruit.csv [part]
puis BASE_CSV=train_v3_bruit.csv python -m donnees.melanger ...
"""

import random
import sys

import pandas as pd

from donnees.normalisation import cle_latine
from evaluation.robustesse_modele import coller, fauter, sans_voyelles

PART = 0.3
GRAINE = 20261127
BRUITS = [("faute", fauter), ("voyelles", sans_voyelles), ("colles", coller)]


def bruiter(train, part=PART, test=None, graine=GRAINE):
    """Copies bruitees d'une part des lignes, un bruit par copie.

    Le bruit est tire au hasard ; s'il ne change rien a la phrase, les suivants
    sont essayes, et une phrase qu'aucun ne change n'est pas copiee.
    """
    choisies = train.sample(frac=part, random_state=graine)
    interdites = set(test["transliteration"].map(cle_latine)) if test is not None else set()
    lignes, ecartees = [], 0
    for x in choisies.itertuples():
        texte = str(x.transliteration)
        tirage = random.Random(f"entrainement|{x.id}")
        ordre = BRUITS[:]
        tirage.shuffle(ordre)
        for nom, bruit in ordre:
            b = bruit(texte, tirage)
            if b != texte:
                break
        else:
            continue
        if cle_latine(b) in interdites:
            ecartees += 1
            continue
        lignes.append({"id": f"bruit-{nom}_{x.id}", "transliteration": b,
                       "darija_norm": x.darija_norm, "source": f"bruit:{nom}"})
    return pd.DataFrame(lignes), ecartees


def main():
    entree, sortie = sys.argv[1], sys.argv[2]
    part = float(sys.argv[3]) if len(sys.argv) > 3 else PART
    from donnees.melanger import TEST, charger

    train = pd.read_csv(entree)
    copies, ecartees = bruiter(train, part, charger(TEST))
    out = pd.concat([train, copies], ignore_index=True)
    out.to_csv(sortie, index=False, encoding="utf-8-sig")
    print(f"{len(train)} lignes, {len(copies)} copies bruitees "
          f"({100 * len(copies) / len(out):.1f} % du resultat), "
          f"{ecartees} ecartees car tombant sur le test")
    print(copies["source"].value_counts().to_string())
    print(f"ecrit dans {sortie} : {len(out)} lignes")


if __name__ == "__main__":
    main()
