"""Ventilation des erreurs de caractere par categorie linguistique.

Un CER global dit qu'une experience a bouge, il ne dit pas ce qu'elle a change.
Ce script repond a la seconde question : il aligne chaque prediction sur sa
reference, classe chaque operation d'edition, et rend la part de chaque
categorie. Lance apres chaque entrainement, il transforme une serie d'essais en
apprentissage.

Le cahier des charges demande cette ventilation explicitement, par consonnes
emphatiques, voyelles, mots d'origine francaise, chiffres et elements non
lexicaux.

La somme des operations classees egale exactement le numerateur du CER rendu
par evaluate.py. Ce n'est pas un detail de presentation : une ventilation qui
ne se reconcilie pas avec la metrique ne prouve rien. L'alignement est donc
calcule par programmation dynamique complete, et non par une heuristique de
plus proche bloc commun, qui rendrait une distance superieure a l'optimale.

Usage :
    python erreurs.py predictions.csv              ventilation
    python erreurs.py apres.csv avant.csv          ecart entre deux experiences
"""

import collections
import sys

import pandas as pd

from evaluate import normaliser, distance

# Voyelles longues. En arabizi elles s'ecrivent de plusieurs facons, ou/u/o
# pour la premiere, a/aa pour la seconde : c'est la ou la variation de graphie
# frappe en premier.
VOYELLES = set("اوي")
# Porteurs de hamza. Leur neutralisation au scoring est arbitree en PO-01.
HAMZAS = set("أإآءئؤ")
# Les quatre emphatiques et leur contrepartie simple, dans les deux sens.
EMPHATIQUES = {"ص": "س", "ض": "د", "ط": "ت", "ظ": "ذ"}
# Le son /g/ n'a pas de lettre arretee en darija : chaque scripteur tranche.
VELAIRES = set("ڭكقگ")
PONCTUATION = set(",،;؛?؟.!:\"'()[]{}-–—…«»")
# Prefixes qui se collent au mot suivant en ecriture arabe alors qu'ils
# s'ecrivent detaches en arabizi : l'article et les prepositions de lieu et de
# moyen. Les ajouter ou les omettre est une decision de frontiere de mot, pas
# une erreur de transcription, et c'est le deuxieme poste d'erreurs du modele.
# Limite assumee : l'alif de l'article ال est compte en voyelle longue, la
# categorie sous-estime donc l'article d'autant.
CLITIQUES = set("لبف")

ORDRE = ["voyelles longues", "espaces et agglutination", "clitiques colles",
         "autres consonnes", "hamzas", "velaires g/k/q",
         "consonnes emphatiques", "ponctuation", "latin residuel", "chiffres"]


def aligner(prediction, reference):
    """Rend la liste des operations d'edition transformant la prediction en reference.

    Chaque operation est un triplet (nature, caractere de la prediction,
    caractere de la reference), l'un des deux etant vide pour une insertion ou
    une omission. La longueur de la liste vaut la distance de Levenshtein.
    """
    a, b = prediction, reference
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cout = 0 if a[i - 1] == b[j - 1] else 1
            table[i][j] = min(table[i - 1][j] + 1,
                              table[i][j - 1] + 1,
                              table[i - 1][j - 1] + cout)

    operations, i, j = [], n, m
    while i > 0 or j > 0:
        cout = 0 if (i > 0 and j > 0 and a[i - 1] == b[j - 1]) else 1
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + cout:
            if cout:
                operations.append(("confusion", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            # caractere present dans la prediction, absent de la reference
            operations.append(("en trop", a[i - 1], ""))
            i -= 1
        else:
            operations.append(("manquant", "", b[j - 1]))
            j -= 1
    return operations[::-1]


def categorie(nature, avant, apres):
    """Classe une operation. L'ordre des tests n'est pas indifferent.

    Les hamzas sont testees avant les voyelles longues, un أ etant un porteur
    de hamza et non un alif ordinaire. Les emphatiques le sont avant les
    consonnes generiques, sans quoi elles s'y noieraient. Les clitiques ne sont
    reconnus que sur un ajout ou une omission : une confusion entre deux
    consonnes ne traduit pas une decision de frontiere de mot.
    """
    paire = {avant, apres} - {""}
    if any(c.isdigit() for c in paire):
        return "chiffres"
    if any(c.isascii() and c.isalpha() for c in paire):
        return "latin residuel"
    if any(c.isspace() for c in paire):
        return "espaces et agglutination"
    if paire & PONCTUATION:
        return "ponctuation"
    if EMPHATIQUES.get(avant) == apres or EMPHATIQUES.get(apres) == avant:
        return "consonnes emphatiques"
    if paire & HAMZAS:
        return "hamzas"
    if paire & VOYELLES:
        return "voyelles longues"
    if nature != "confusion" and paire <= CLITIQUES:
        return "clitiques colles"
    if paire and paire <= VELAIRES:
        return "velaires g/k/q"
    return "autres consonnes"


def ventiler(donnees):
    """Compte et classe toutes les operations d'un fichier de predictions."""
    comptes = collections.Counter()
    natures = collections.defaultdict(collections.Counter)
    exemples = collections.defaultdict(collections.Counter)
    total_distance = 0

    for reference, prediction in zip(donnees["darija_norm"], donnees["prediction"]):
        reference = normaliser(str(reference))
        prediction = normaliser(str(prediction))
        total_distance += distance(prediction, reference)
        for nature, avant, apres in aligner(prediction, reference):
            c = categorie(nature, avant, apres)
            comptes[c] += 1
            natures[c][nature] += 1
            exemples[c][f"{avant or '_'} vers {apres or '_'}"] += 1

    # Garde-fou : une ventilation qui ne se reconcilie pas avec la metrique ne
    # demontre rien. L'egalite doit etre exacte, pas approchee.
    assert sum(comptes.values()) == total_distance, (
        f"{sum(comptes.values())} operations classees contre {total_distance} "
        "comptees par evaluate.py")
    return comptes, natures, exemples, total_distance


def afficher(comptes, natures, exemples, total, libelle):
    print(f"\n{libelle} : {total} operations d'edition")
    print(f"{'categorie':<26} {'ops':>6} {'part':>7} {'en trop':>9} "
          f"{'manquant':>9} {'confusion':>10}   principaux cas")
    for c in ORDRE:
        if not comptes[c]:
            continue
        n = natures[c]
        top = ", ".join(f"{k} ({v})" for k, v in exemples[c].most_common(2))
        print(f"{c:<26} {comptes[c]:>6} {100 * comptes[c] / total:>6.1f}% "
              f"{n['en trop']:>9} {n['manquant']:>9} {n['confusion']:>10}   {top}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[-2].strip())

    actuel = pd.read_csv(sys.argv[1])
    c1, n1, e1, t1 = ventiler(actuel)
    afficher(c1, n1, e1, t1, sys.argv[1])

    if len(sys.argv) > 2:
        precedent = pd.read_csv(sys.argv[2])
        c0, _, _, t0 = ventiler(precedent)
        print(f"\nEcart avec {sys.argv[2]} : {t0} operations contre {t1}, "
              f"soit {100 * (t1 - t0) / max(t0, 1):+.1f}%")
        print(f"{'categorie':<26} {'avant':>8} {'apres':>8} {'ecart':>8} "
              f"{'part de l_ecart':>16}")
        delta_total = t1 - t0
        for c in ORDRE:
            if not (c0[c] or c1[c]):
                continue
            d = c1[c] - c0[c]
            part = f"{100 * d / delta_total:>14.1f} %" if delta_total else " " * 16
            print(f"{c:<26} {c0[c]:>8} {c1[c]:>8} {d:>+8} {part}")
        print("\n  Une experience utile deplace une categorie visee, pas "
              "seulement le total.\n  Un gain reparti uniformement sur toutes "
              "les categories signale un effet\n  de volume ou de duree, non le "
              "mecanisme qu'on croyait traiter.")
