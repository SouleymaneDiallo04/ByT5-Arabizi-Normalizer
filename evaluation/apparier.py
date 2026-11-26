"""Deux modeles, phrase par phrase, sur les memes phrases de TerjamaBench.

Un CER global bouge aussi par hasard : sur 678 phrases, un ecart de trois
milliemes reste dans le bruit (journal du 21 novembre). Ce script compare deux
fichiers de predictions phrase par phrase, sur le reglage et sur l'evaluation :
phrases exactes gagnees et perdues, phrases meilleures et pires, chacune avec
le test du signe ; puis par variete regionale et par theme.

Il compte aussi les sorties restees en latin, et les caracteres que le modele
ajoute ou oublie le plus : espaces, ل et ا (journal du 21 novembre).
Avec --sans-latin, la comparaison se restreint aux phrases ou aucune des deux
sorties n'a de latin. Ce n'est qu'un diagnostic : la selection depend des
sorties, et une phrase recopiee en latin par un modele en sort.

La ventilation par categorie se lit avec :
    python -m evaluation.erreurs apres.csv avant.csv

Usage, depuis la racine du depot :
    python -m evaluation.apparier apres.csv avant.csv
    python -m evaluation.apparier apres.csv avant.csv --sans-latin
"""

import argparse
import collections
import math
import re

import pandas as pd

from donnees.decouper_test import prelever
from evaluation.erreurs import aligner
from evaluation.evaluate import mesurer, normaliser
from evaluation.relecture import par_phrase

LATIN = re.compile(r"[A-Za-z]")
SURVEILLES = (" ", "ل", "ا")


def signe(gagnees, perdues):
    """Test du signe bilateral exact : p de voir un ecart au moins aussi grand
    si chaque phrase avait une chance sur deux de pencher d'un cote."""
    n = gagnees + perdues
    if not n:
        return 1.0
    k = min(gagnees, perdues)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def a_du_latin(texte):
    return bool(LATIN.search(normaliser(str(texte))))


def comparer(d):
    """Metriques des deux modeles et bilan apparie, sur un cadre a colonnes
    darija_norm, avant et apres."""
    ref = d["darija_norm"].map(normaliser)
    a, b = par_phrase(ref, d["avant"]), par_phrase(ref, d["apres"])
    ma = mesurer(ref, d["avant"].map(normaliser))
    mb = mesurer(ref, d["apres"].map(normaliser))
    gagnees = int((b["exacte"] & ~a["exacte"]).sum())
    perdues = int((~b["exacte"] & a["exacte"]).sum())
    mieux = int((b["erreurs"] < a["erreurs"]).sum())
    pire = int((b["erreurs"] > a["erreurs"]).sum())
    return {
        "phrases": len(d),
        "cer": (ma["cer"], mb["cer"]), "wer": (ma["wer"], mb["wer"]),
        "exactes": (int(a["exacte"].sum()), int(b["exacte"].sum())),
        "latin": (int(d["avant"].map(a_du_latin).sum()),
                  int(d["apres"].map(a_du_latin).sum())),
        "gagnees": gagnees, "perdues": perdues, "p_exactes": signe(gagnees, perdues),
        "mieux": mieux, "pire": pire, "p_phrases": signe(mieux, pire),
    }


def asymetries(references, predictions, caracteres=SURVEILLES):
    """Pour chaque caractere, combien de fois il est en trop ou manquant."""
    comptes = collections.Counter()
    for r, p in zip(references, predictions):
        for nature, avant, apres in aligner(normaliser(str(p)), normaliser(str(r))):
            if nature == "en trop" and avant in caracteres:
                comptes[(avant, "en trop")] += 1
            elif nature == "manquant" and apres in caracteres:
                comptes[(apres, "manquant")] += 1
    return comptes


def afficher(nom, c):
    print(f"\n== {nom}, {c['phrases']} phrases")
    print(f"{'':8}{'CER':>9}{'WER':>9}{'exactes':>9}{'latin':>7}")
    for i, lib in enumerate(("avant", "apres")):
        print(f"{lib:8}{c['cer'][i]:>9.4f}{c['wer'][i]:>9.4f}"
              f"{c['exactes'][i]:>9}{c['latin'][i]:>7}")
    print(f"exactes gagnees / perdues : {c['gagnees']} / {c['perdues']}, "
          f"p = {c['p_exactes']:.3f}")
    print(f"phrases meilleures / pires : {c['mieux']} / {c['pire']}, "
          f"p = {c['p_phrases']:.3f}")


def par_groupe(d, colonne):
    print(f"\n== evaluation, par {colonne}")
    print(f"{'':22}{'phrases':>8}{'CER avant':>11}{'CER apres':>11}"
          f"{'meilleures':>12}{'pires':>7}{'latin apres':>13}")
    for valeur, x in sorted(d.groupby(colonne), key=lambda kv: -len(kv[1])):
        c = comparer(x)
        print(f"{str(valeur)[:22]:22}{c['phrases']:>8}{c['cer'][0]:>11.4f}"
              f"{c['cer'][1]:>11.4f}{c['mieux']:>12}{c['pire']:>7}{c['latin'][1]:>13}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("apres", help="predictions du modele a juger")
    p.add_argument("avant", help="predictions du modele de reference")
    p.add_argument("--sans-latin", action="store_true",
                   help="seulement les phrases ou aucune sortie n'a de latin")
    a = p.parse_args()

    apres, avant = pd.read_csv(a.apres), pd.read_csv(a.avant)
    if apres["id"].tolist() != avant["id"].tolist():
        raise SystemExit("les deux fichiers ne portent pas les memes phrases")
    d = apres.rename(columns={"prediction": "apres"}).assign(avant=avant["prediction"])
    # Le prelevement se fait sur les 850 phrases, avant tout filtre : il est
    # stratifie et seme, et ne retomberait pas sur les memes phrases sinon.
    reglage, evaluation = prelever(d)
    if a.sans_latin:
        garder = lambda x: x[~(x["avant"].map(a_du_latin) | x["apres"].map(a_du_latin))]
        d, reglage, evaluation = garder(d), garder(reglage), garder(evaluation)
        print("Phrases ou aucune des deux sorties n'a de latin : diagnostic seulement.")

    afficher("reglage", comparer(reglage))
    afficher("evaluation", comparer(evaluation))
    par_groupe(evaluation, "annotator_dialect")
    par_groupe(evaluation, "topic")

    print(f"\n== caracteres en trop ou manquants, sur les {len(d)} phrases")
    av = asymetries(d["darija_norm"], d["avant"])
    ap = asymetries(d["darija_norm"], d["apres"])
    for car in SURVEILLES:
        for nature in ("en trop", "manquant"):
            k = (car, nature)
            print(f"  {car!r:>5} {nature:<10}{av[k]:>6}{ap[k]:>6}{ap[k] - av[k]:>+6}")

    latines = d[d["avant"].map(a_du_latin) | d["apres"].map(a_du_latin)]
    if len(latines):
        print(f"\n== {len(latines)} phrases ou une sortie garde du latin")
        for _, x in latines.iterrows():
            print(f"  {x['id']}  {x['transliteration']}\n    avant : {x['avant']}"
                  f"\n    apres : {x['apres']}")


if __name__ == "__main__":
    main()
