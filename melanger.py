"""Melange du corpus d'entrainement avec un corpus candidat, a budget fixe.

Pourquoi un script et non quelques lignes dans un carnet. Trois operations
doivent etre faites et aucune ne se devine : purger les fuites vers le jeu de
test, calculer la cible normalisee avec la convention de make_splits.py, et
fixer la proportion du melange. Les trois sont semees, donc rejouables, ce
qu'exige ENF-04.

Le fichier produit compte exactement le nombre de lignes que l'entrainement
consommera en un passage : 8 760 pas de 64 exemples font 560 640 lignes. Une
epoque sur ce fichier vaut donc exactement le budget de la reference, et la
proportion ecrite ici est celle que le modele verra, sans qu'un tirage du
Trainer vienne la deformer.

La base est sur-echantillonnee plutot que tiree avec remise : le corpus entier
figure d'abord une fois, puis un tirage complete jusqu'au compte voulu. Chaque
paire est ainsi vue au moins une fois, ce qu'un tirage avec remise ne garantit
pas.

ATTENTION. Le corpus candidat peut recouper le jeu de test, et rien ne le
garantit a l'avance. La purge est faite ici sur les deux cles de
make_splits.py, l'arabe et l'arabizi, et le nombre de lignes retirees est
affiche. Sans elle, l'evaluation ne vaudrait plus rien.

Usage :
    python melanger.py nilechat_translit.csv
    python melanger.py nilechat_translit.csv 0.5 8760
"""

import sys

import pandas as pd
from huggingface_hub import hf_hub_download

from make_splits import normaliser_cible, cle_arabe, cle_latine

DEPOT = "sawti-nlp/darija-arabizi-corpus"
BASE, TEST = "train.csv", "test_real.csv"
SORTIE = "train_melange.csv"

# 8 760 pas de 16 exemples par carte, sur deux cartes, avec deux pas
# d'accumulation : c'est le budget exact du run de reference.
PAS = 8760
LOT_EFFECTIF = 64
PART_BASE = 0.5
GRAINE = 42
COLONNES = ["id", "transliteration", "darija_norm", "source"]


def charger(chemin):
    try:
        return pd.read_csv(chemin)
    except FileNotFoundError:
        return pd.read_csv(hf_hub_download(DEPOT, f"splits/{chemin}",
                                           repo_type="dataset"))


def preparer(candidat, test):
    """Purge les fuites vers le jeu de test et calcule la cible normalisee."""
    avant = len(candidat)
    candidat = candidat.dropna(subset=["darija", "transliteration"])
    candidat = candidat[(candidat["darija"].astype(str).str.strip() != "")
                        & (candidat["transliteration"].astype(str).str.strip() != "")]
    vides = avant - len(candidat)

    ka = candidat["darija"].map(cle_arabe)
    kl = candidat["transliteration"].map(cle_latine)
    fuite = ka.isin(set(test["darija"].map(cle_arabe))) | \
        kl.isin(set(test["transliteration"].map(cle_latine)))
    candidat = candidat[~fuite]

    # Doublons internes sur le couple des deux cles : trois millions de lignes
    # tirees de documents web en contiennent forcement.
    candidat = candidat.assign(_k=ka[~fuite] + "||" + kl[~fuite])
    avant_doublons = len(candidat)
    candidat = candidat.drop_duplicates("_k").drop(columns="_k")

    candidat = candidat.assign(darija_norm=candidat["darija"].map(normaliser_cible))
    return candidat, vides, int(fuite.sum()), avant_doublons - len(candidat)


def tirer(donnees, combien, graine):
    """Rend exactement `combien` lignes, en montrant chaque ligne au moins une fois."""
    if combien <= len(donnees):
        return donnees.sample(combien, random_state=graine)
    reste = combien - len(donnees)
    tours = [donnees]
    while reste > 0:
        part = donnees.sample(min(reste, len(donnees)), random_state=graine + len(tours))
        tours.append(part)
        reste -= len(part)
    return pd.concat(tours, ignore_index=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage : python melanger.py corpus_candidat.csv "
                         "[part_base] [pas]")
    chemin = sys.argv[1]
    part = float(sys.argv[2]) if len(sys.argv) > 2 else PART_BASE
    pas = int(sys.argv[3]) if len(sys.argv) > 3 else PAS

    total = pas * LOT_EFFECTIF
    n_base = round(total * part)
    n_candidat = total - n_base

    base = charger(BASE)
    test = charger(TEST)
    candidat = pd.read_csv(chemin)
    print(f"base      {BASE:<24} {len(base):>9} paires")
    print(f"candidat  {chemin[:24]:<24} {len(candidat):>9} paires")

    candidat, vides, fuites, doublons = preparer(candidat, test)
    print(f"\npurge du candidat")
    print(f"  cotes vides            {vides:>9}")
    print(f"  fuites vers le test    {fuites:>9}")
    print(f"  doublons internes      {doublons:>9}")
    print(f"  reste                  {len(candidat):>9}")

    if n_candidat > len(candidat):
        raise SystemExit(f"Le candidat ne compte que {len(candidat)} paires "
                         f"apres purge, il en faut {n_candidat}.")

    melange = pd.concat([
        tirer(base[COLONNES], n_base, GRAINE),
        tirer(candidat[COLONNES], n_candidat, GRAINE),
    ], ignore_index=True).sample(frac=1, random_state=GRAINE).reset_index(drop=True)
    melange.to_csv(SORTIE, index=False, encoding="utf-8-sig")

    print(f"\n{SORTIE} : {len(melange)} lignes pour {pas} pas de {LOT_EFFECTIF}")
    print(f"  base      {n_base:>9} lignes, soit {n_base / len(base):.2f} passage(s) "
          f"sur le corpus actuel")
    print(f"  candidat  {n_candidat:>9} lignes, soit "
          f"{100 * n_candidat / len(candidat):.1f} % du candidat")
    print(f"\nLancer :  TRAIN_CSV={SORTIE} MAX_PAS={pas} "
          f"SORTIE_PREDICTIONS=byt5_melange.csv python train_byt5.py")
