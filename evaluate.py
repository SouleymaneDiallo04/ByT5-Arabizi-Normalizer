"""Calcul des metriques d'evaluation, commande unique et reproductible.

Ce script est le seul point de mesure du projet : la baseline, le modele ByT5 et
les modeles publics auxquels nous nous comparons passent tous par lui. Si chacun
calculait son propre CER, les chiffres ne seraient pas comparables entre eux.

Usage : python evaluate.py [fichier_de_predictions.csv]
"""

import re
import sys
import unicodedata

import pandas as pd

PREDICTIONS = sys.argv[1] if len(sys.argv) > 1 else "baseline_predictions.csv"

# Meme normalisation que make_splits.py. Toute modification doit etre reportee
# dans les deux fichiers, sans quoi les scores cessent d'etre comparables.
VOCALISATION = [chr(c) for c in range(0x064B, 0x0656)] + ["\u0640", "\u0670"]
TABLE_VOCAL = {ord(c): None for c in VOCALISATION}
PONCTUATION_ARABE = {ord("?"): "؟", ord(";"): "؛"}


def normaliser(texte):
    """Ramene un texte a la convention retenue : sans vocalisation, ponctuation arabe.

    Le cahier des charges impose d'appliquer cette normalisation aux references
    comme aux predictions avant tout calcul, faute de quoi le modele serait
    penalise pour des differences qui n'en sont pas.
    """
    t = unicodedata.normalize("NFKC", str(texte)).translate(TABLE_VOCAL)
    return re.sub(r"\s+", " ", t.translate(PONCTUATION_ARABE)).strip()


def distance(a, b):
    """Distance de Levenshtein entre deux chaines, deux lignes en memoire."""
    if a == b:
        return 0
    precedente = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        courante = [i]
        for j, cb in enumerate(b, 1):
            courante.append(min(precedente[j] + 1,
                                courante[j - 1] + 1,
                                precedente[j - 1] + (ca != cb)))
        precedente = courante
    return precedente[-1]


def mesurer(references, predictions):
    """Calcule le CER et l'exactitude phrase sur deux series alignees.

    Le CER est calcule au niveau du corpus, somme des distances sur somme des
    longueurs de reference. La moyenne des CER par phrase donnerait un poids
    excessif aux phrases courtes, dont le jeu de test contient beaucoup.
    """
    erreurs = longueurs = exactes = 0
    for reference, prediction in zip(references, predictions):
        reference, prediction = str(reference), str(prediction)
        erreurs += distance(prediction, reference)
        longueurs += len(reference)
        exactes += (prediction == reference)
    nombre = max(len(references), 1)
    return {"cer": erreurs / max(longueurs, 1),
            "exactitude": exactes / nombre,
            "phrases": nombre}


def ventiler(donnees, colonne, libelle):
    """Affiche les metriques groupees par une colonne, du groupe le plus fourni au moins."""
    print(f"\n{libelle}")
    print(f"{'':22} {'phrases':>8} {'CER':>8} {'exactitude':>12}")
    for valeur, part in sorted(donnees.groupby(colonne), key=lambda kv: -len(kv[1])):
        m = mesurer(part["darija_norm"], part["pred_norm"])
        print(f"{str(valeur)[:22]:22} {m['phrases']:>8} {m['cer']:>8.4f} "
              f"{100 * m['exactitude']:>11.1f}%")


if __name__ == "__main__":
    try:
        d = pd.read_csv(PREDICTIONS)
    except FileNotFoundError:
        raise SystemExit(f"{PREDICTIONS} absent : produire d'abord les predictions")

    manquantes = {"prediction", "darija", "darija_norm"} - set(d.columns)
    if manquantes:
        raise SystemExit(f"colonnes absentes de {PREDICTIONS} : {sorted(manquantes)}")

    d["prediction"] = d["prediction"].fillna("")
    d["pred_norm"] = d["prediction"].map(normaliser)

    brut = mesurer(d["darija"], d["prediction"])
    norme = mesurer(d["darija_norm"], d["pred_norm"])

    print(f"{PREDICTIONS} : {len(d)} phrases\n")
    print(f"{'reference':36} {'CER':>8} {'exactitude':>12}")
    print("-" * 58)
    print(f"{'texte brut':36} {brut['cer']:>8.4f} {100 * brut['exactitude']:>11.1f}%")
    print(f"{'convention appliquee des deux cotes':36} {norme['cer']:>8.4f} "
          f"{100 * norme['exactitude']:>11.1f}%")
    print("-" * 58)
    print(f"ecart de CER imputable a la convention : {brut['cer'] - norme['cer']:+.4f}")

    if "mot_isole" in d.columns:
        print("\nMots isoles et phrases, mesures separement")
        print(f"{'':22} {'phrases':>8} {'CER':>8} {'exactitude':>12}")
        for libelle, part in [("mots isoles", d[d["mot_isole"]]),
                              ("phrases", d[~d["mot_isole"]])]:
            if len(part):
                m = mesurer(part["darija_norm"], part["pred_norm"])
                print(f"{libelle:22} {m['phrases']:>8} {m['cer']:>8.4f} "
                      f"{100 * m['exactitude']:>11.1f}%")

    if "annotator_dialect" in d.columns:
        ventiler(d, "annotator_dialect", "Par variete regionale de l'annotateur")
    if "topic" in d.columns:
        ventiler(d, "topic", "Par theme")
