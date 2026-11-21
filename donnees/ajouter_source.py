"""Ajoute une source au jeu d'entrainement, sans toucher au dev ni au test.

Pourquoi ne pas relancer make_splits.py. Il tire le dev au hasard parmi toutes
les phrases du corpus : une seule ligne ajoutee decale le tirage et renouvelle
tout le dev. Des phrases deja vues a l'entrainement y entreraient, les modeles
existants seraient evalues sur du deja vu, et les pertes de validation ne se
compareraient plus d'une experience a l'autre.

Ici, train.csv est recopie tel quel et les nouvelles paires s'ajoutent a la
suite. Une paire n'entre que si :

  - ni sa phrase arabe ni son arabizi ne figure dans le jeu de test ;
  - ni l'une ni l'autre ne figure dans le dev ;
  - la paire n'est pas deja dans l'entrainement.

Le nettoyage, les cles et la cible normalisee sont ceux de merge_all.py et de
make_splits.py. Les drapeaux `conflit` et `suspect` sont calcules pour les
nouvelles lignes et ne suppriment rien, comme dans le corpus d'origine. Ceux des
lignes existantes ne sont pas recalcules : le fichier d'origine reste intact.

Usage, depuis le dossier des donnees :
    python -m donnees.ajouter_source doda_audio_translit.csv train_v2.csv
"""

import sys

import pandas as pd

from donnees.make_splits import cle_arabe, cle_latine, normaliser_cible
from donnees.melanger import charger
from donnees.merge_all import ARTEFACTS, RATIO_MAX, RATIO_MIN, nettoyer, richesse

COLONNES_SOURCE = ["id", "darija", "transliteration", "source"]


def integrer(candidat, train, dev, test):
    """Renvoie les lignes a ajouter a `train`, et le bilan de ce qui est ecarte.

    Chaque ligne ecartee l'est pour un seul motif, le premier rencontre : la
    somme du bilan et des lignes gardees redonne donc la taille du candidat.
    """
    bilan = {"lues": len(candidat)}

    c = candidat[COLONNES_SOURCE].dropna(subset=["darija", "transliteration"]).copy()
    c["darija"] = c["darija"].map(nettoyer)
    c["transliteration"] = c["transliteration"].map(nettoyer)
    c["key_ar"] = c["darija"].map(cle_arabe)
    c["key_la"] = c["transliteration"].map(cle_latine)
    # Une ligne faite de ponctuation a une cle vide, et se confondrait avec
    # toutes les autres dans ce cas.
    c = c[(c["key_ar"] != "") & (c["key_la"] != "")]
    bilan["vides"] = bilan["lues"] - len(c)

    def retirer(masque, motif):
        nonlocal c
        bilan[motif] = int(masque.sum())
        c = c[~masque]

    retirer(c["darija"].str.contains(ARTEFACTS, na=False), "consignes de generation")
    retirer(c["key_ar"].isin(set(test["key_ar"])) | c["key_la"].isin(set(test["key_la"])),
            "fuites vers le test")
    retirer(c["key_ar"].isin(set(dev["key_ar"])) | c["key_la"].isin(set(dev["key_la"])),
            "fuites vers le dev")
    deja = set(zip(train["key_ar"], train["key_la"]))
    retirer(pd.Series([p in deja for p in zip(c["key_ar"], c["key_la"])], index=c.index, dtype=bool),
            "deja dans l'entrainement")

    # Doublons internes : on garde la copie vocalisee, comme merge_all.py.
    avant = len(c)
    c = (c.assign(_richesse=c["darija"].map(richesse))
         .sort_values("_richesse", ascending=False, kind="stable")
         .drop_duplicates(subset=["key_ar", "key_la"], keep="first")
         .drop(columns="_richesse")
         .sort_index())
    bilan["doublons internes"] = avant - len(c)

    c["darija_norm"] = c["darija"].map(normaliser_cible)
    ratio = c["key_ar"].str.len() / c["key_la"].str.len().clip(lower=1)
    c["suspect"] = (ratio < RATIO_MIN) | (ratio > RATIO_MAX)
    cibles = pd.concat([train[["key_la", "key_ar"]], c[["key_la", "key_ar"]]])
    ambigues = cibles.groupby("key_la")["key_ar"].nunique()
    c["conflit"] = c["key_la"].map(ambigues) > 1

    bilan["ajoutees"] = len(c)
    return c[list(train.columns)], bilan


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage : python -m donnees.ajouter_source candidat.csv sortie.csv")
    chemin, sortie = sys.argv[1], sys.argv[2]

    train, dev, test = charger("train.csv"), charger("dev.csv"), charger("test_real.csv")
    nouveaux, bilan = integrer(pd.read_csv(chemin), train, dev, test)

    for motif, n in bilan.items():
        print(f"  {motif:28} {n:>7}")
    print(f"  dont marquees conflit        {int(nouveaux['conflit'].sum()):>7}")
    print(f"  dont marquees suspect        {int(nouveaux['suspect'].sum()):>7}")

    complet = pd.concat([train, nouveaux], ignore_index=True)
    # Controles sur le fichier final, pas seulement sur les lignes ajoutees.
    if not complet.iloc[:len(train)]["id"].equals(train["id"].reset_index(drop=True)):
        raise SystemExit("ANOMALIE : les lignes d'origine ont bouge.")
    for nom, autre in (("dev", dev), ("test", test)):
        if set(nouveaux["key_ar"]) & set(autre["key_ar"]):
            raise SystemExit(f"ANOMALIE : une phrase ajoutee figure dans le {nom}.")

    complet.to_csv(sortie, index=False, encoding="utf-8-sig")
    print(f"\n{sortie} : {len(train)} lignes d'origine + {len(nouveaux)} = {len(complet)}")
    print(complet["source"].value_counts().to_string())
