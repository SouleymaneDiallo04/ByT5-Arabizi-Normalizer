"""Augmentation de l'entree par graphies arabizi attestees.

Axe 1 de l'iteration 2. L'hypothese tient en deux mesures.

Primo, 99,42 % des phrases d'entrainement n'existent qu'avec une seule graphie
arabizi, alors que le jeu de test est ecrit par des humains qui, eux, varient.
Le modele n'a donc jamais vu deux ecritures d'une meme phrase.

Secundo, la ventilation des erreurs attribue 43,5 % des operations restantes
aux voyelles longues, et 24,1 % aux frontieres de mots. Ce sont exactement les
deux choses qui changent d'un scripteur a l'autre.

Attention a ne pas surinterpreter : le corpus n'est pas depourvu de variation.
Au niveau du mot, un mot arabe frequent s'ecrit de 2 facons en mediane. Ce qui
manque n'est pas la variation mais sa repartition : la graphie dominante ecrase
les autres, et aucune phrase n'apparait deux fois autrement ecrite.

Ce script ne fabrique aucune regle. Il releve les graphies que le corpus
atteste deja pour chaque mot, puis les permute. Une variante ne contient donc
que des ecritures que quelqu'un a reellement produites.

Usage :
    python augmenter.py             2 variantes par phrase, soit un corpus x3
    python augmenter.py 1           une seule variante, corpus x2
"""

import collections
import random
import re
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

from baseline import mots
from evaluate import distance

TRAIN = "train.csv"
SORTIE = "train_augmente.csv"
DEPOT = "sawti-nlp/darija-arabizi-corpus"

# Un mot arabe doit etre assez vu pour que ses graphies ne soient pas des
# accidents d'alignement.
MIN_OCCURRENCES = 20
# Une graphie doit l'etre aussi, pour la meme raison.
MIN_VARIANTE = 3
# Ecart maximal a la graphie dominante, en part de sa longueur. L'alignement
# positionnel se trompe sur les paires agglutinees et attribue a un mot la
# graphie de son voisin : sans ce garde-fou, l'inventaire contiendrait ta3 pour
# dyal. Mesure : il ramene la mediane de 8 graphies par mot a 2.
RATIO_DISTANCE = 0.4
# Une graphie doit designer un seul mot arabe. la vaut pour 3la comme pour la,
# la permuter melangerait deux mots au lieu de varier l'ecriture d'un seul.
PURETE = 0.8
# Part des mots eligibles effectivement permutes dans une variante. Tout
# permuter produirait une phrase ou chaque mot est ecrit de facon rare, ce
# qu'aucun scripteur ne fait : chacun a ses habitudes et en devie par endroits.
PROBA_SUBSTITUTION = 0.35
TENTATIVES = 4
GRAINE = 42

DECOUPE = re.compile(r"([^\w]+)")


def inventaire(donnees):
    """Releve, pour chaque graphie arabizi, les autres graphies du meme mot arabe.

    L'appariement des mots est positionnel et ne vaut que sur les paires dont
    les deux cotes comptent le meme nombre de mots, l'agglutination arabe
    rendant les autres inalignables.
    """
    formes = collections.defaultdict(collections.Counter)   # mot arabe -> graphies
    sens = collections.defaultdict(collections.Counter)     # graphie -> mots arabes

    for latin, arabe in zip(donnees["transliteration"].astype(str),
                            donnees["darija"].astype(str)):
        gauche, droite = mots(latin), mots(arabe)
        if len(gauche) == len(droite) and gauche:
            for g, d in zip(gauche, droite):
                formes[d][g.lower()] += 1
                sens[g.lower()][d] += 1

    table, retenus = collections.defaultdict(list), 0
    for arabe, graphies in formes.items():
        if sum(graphies.values()) < MIN_OCCURRENCES:
            continue
        dominante = graphies.most_common(1)[0][0]
        plausibles = [dominante]
        for graphie, n in graphies.items():
            if graphie == dominante or n < MIN_VARIANTE:
                continue
            if distance(graphie, dominante) / max(len(dominante), 1) > RATIO_DISTANCE:
                continue
            plausibles.append(graphie)
        # La purete s'applique avant de construire les alternatives, et non en
        # filtrant les seules cles. Sans cela une graphie ambigue, entree par
        # erreur d'alignement, resterait proposee en remplacement d'une graphie
        # saine : bach se voyait offrir wach, qui est un autre mot.
        purs = []
        for graphie in plausibles:
            total = sum(sens[graphie].values())
            if total and sens[graphie][arabe] / total >= PURETE:
                purs.append(graphie)
        if len(purs) < 2:
            continue
        retenus += 1
        for graphie in purs:
            table[graphie] = [a for a in purs if a != graphie]
    return table, retenus


def varier(phrase, table, tirage):
    """Rend une autre ecriture de la meme phrase, ou None si rien n'a change.

    La ponctuation, les chiffres isoles et les mots inconnus de l'inventaire
    sont laisses intacts : on ne varie que ce que le corpus atteste.
    """
    morceaux = DECOUPE.split(str(phrase))
    change = False
    for i, morceau in enumerate(morceaux):
        autres = table.get(morceau.lower())
        if not autres or tirage.random() > PROBA_SUBSTITUTION:
            continue
        choisi = tirage.choice(autres)
        if morceau[:1].isupper():
            choisi = choisi.capitalize()
        morceaux[i] = choisi
        change = True
    return "".join(morceaux) if change else None


if __name__ == "__main__":
    variantes = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    try:
        train = pd.read_csv(TRAIN)
    except FileNotFoundError:
        # Sur une machine jetable comme Kaggle, train.csv n'est pas la : il se
        # recupere depuis le depot du projet, comme le fait train_byt5.py.
        train = pd.read_csv(hf_hub_download(DEPOT, f"splits/{TRAIN}",
                                            repo_type="dataset"))

    table, retenus = inventaire(train)
    couverture = sum(len(v) for v in table.values()) / max(len(table), 1)
    print(f"{retenus} mots arabes offrent plusieurs graphies plausibles")
    print(f"{len(table)} graphies permutables, {couverture:.1f} alternatives "
          f"en moyenne")

    tirage = random.Random(GRAINE)
    lignes, echecs = [], 0
    for _, ligne in train.iterrows():
        # Une variante identique a l'originale, ou a une variante deja produite
        # pour la meme phrase, ne serait qu'un doublon de plus a entrainer.
        vues = {str(ligne["transliteration"])}
        for rang in range(1, variantes + 1):
            nouvelle = None
            for _ in range(TENTATIVES):
                essai = varier(ligne["transliteration"], table, tirage)
                if essai and essai not in vues:
                    nouvelle = essai
                    break
            if nouvelle is None:
                echecs += 1
                continue
            vues.add(nouvelle)
            copie = ligne.copy()
            copie["id"] = f"{ligne['id']}_v{rang}"
            copie["transliteration"] = nouvelle
            lignes.append(copie)

    augmente = pd.concat([train, pd.DataFrame(lignes)], ignore_index=True)
    augmente.to_csv(SORTIE, index=False, encoding="utf-8-sig")

    attendu = len(train) * variantes
    print(f"\n{len(lignes)} variantes produites sur {attendu} demandees "
          f"({100 * echecs / max(attendu, 1):.1f} % de phrases sans mot "
          f"permutable)")
    print(f"{SORTIE} : {len(augmente)} lignes, soit x{len(augmente) / len(train):.2f}")
    print(f"la cible n'est jamais touchee : seule l'entree varie")

    print("\nexemples, original puis variantes :")
    for identifiant in train["id"].head(200).sample(3, random_state=GRAINE):
        print(f"  {train.loc[train['id'] == identifiant, 'transliteration'].iloc[0]}")
        for _, v in augmente[augmente["id"].astype(str)
                             .str.startswith(f"{identifiant}_v")].iterrows():
            print(f"    -> {v['transliteration']}")
