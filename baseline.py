"""Baseline deterministe de normalisation arabizi vers ecriture arabe (EF-05).

Deux etages, dans cet ordre : une table de correspondance de mots apprise sur le
corpus d'entrainement, puis une conversion caractere par caractere pour les mots
absents de cette table. Rien n'est entraine, aucun parametre n'est ajuste.

Les fonctions sont importables sans effet de bord ; le pipeline complet ne
s'execute que si le fichier est lance directement.
"""

import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

TRAIN = "train.csv"
TEST = "test_real.csv"
OUTPUT = "baseline_predictions.csv"
# Un couple de mots n'entre dans la table qu'au-dela de ce nombre d'observations,
# et s'il represente au moins cette part des traductions vues pour ce mot latin.
MIN_OBSERVATIONS = 3
PART_MINIMALE = 0.5

# Les digrammes passent avant les lettres seules, sinon "ch" donnerait "ك ه"
DIGRAMMES = [
    ("ch", "ش"), ("sh", "ش"), ("kh", "خ"), ("gh", "غ"),
    ("th", "ث"), ("dh", "ذ"), ("ou", "و"), ("aa", "ا"), ("ee", "ي"),
]
CHIFFRES = {"2": "ء", "3": "ع", "5": "خ", "6": "ط", "7": "ح", "8": "ه", "9": "ق"}
LETTRES = {
    "a": "ا", "b": "ب", "c": "ك", "d": "د", "e": "", "f": "ف", "g": "ڭ",
    "h": "ه", "i": "ي", "j": "ج", "k": "ك", "l": "ل", "m": "م", "n": "ن",
    "o": "و", "p": "پ", "q": "ق", "r": "ر", "s": "س", "t": "ت", "u": "و",
    "v": "ف", "w": "و", "x": "كس", "y": "ي", "z": "ز",
}
PONCTUATION = "،,.!?؟:;\"«»()[]…-'"
ARTICLE = "ال"
# Formes sous lesquelles l'article defini apparait en arabizi
PREFIXES_ARTICLE = ("l", "al", "el", "la", "le")


def mots(texte):
    """Decoupe un texte en mots, ponctuation de bord retiree."""
    return [t for t in (x.strip(PONCTUATION) for x in str(texte).split()) if t]


def par_regles(mot):
    """Convertit un mot latin en ecriture arabe, caractere par caractere.

    Les accents francais sont retires d'abord : sans cela ils traversent la
    conversion et ressortent tels quels dans la sortie arabe (theorie).
    Un chiffre forme a lui seul un mot reste un nombre, conformement a EF-02 ;
    colle a des lettres, il represente une consonne.
    """
    t = unicodedata.normalize("NFKD", str(mot).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    for avant, apres in DIGRAMMES:
        t = t.replace(avant, apres)
    isole = len(str(mot)) == 1
    return "".join(c if c.isdigit() and isole
                   else CHIFFRES.get(c, c) if c.isdigit()
                   else LETTRES.get(c, c)
                   for c in t)


def construire_table(train):
    """Apprend une table mot latin vers mot arabe sur le corpus d'entrainement.

    L'alignement est positionnel, donc limite aux paires dont les deux cotes ont
    le meme nombre de mots : c'est le seul alignement fiable sans outil dedie.
    L'arabe agglutine ses prefixes, ce qui ecarte environ 63 % des paires.

    L'article defini est retire de la forme arabe quand l'arabizi ne le porte
    pas : sans cela la table apprend la forme la plus frequente du corpus et
    colle un article de trop, ce qui fait echouer le cas de test d'EF-02.
    """
    couples = Counter()
    for latin, arabe in zip(train["key_la"], train["darija_norm"]):
        gauche, droite = mots(latin), mots(arabe)
        if len(gauche) != len(droite) or not gauche:
            continue
        for mot_latin, mot_arabe in zip(gauche, droite):
            mot_latin = mot_latin.lower()
            if (mot_arabe.startswith(ARTICLE) and len(mot_arabe) > 3
                    and not mot_latin.startswith(PREFIXES_ARTICLE)):
                mot_arabe = mot_arabe[len(ARTICLE):]
            couples[(mot_latin, mot_arabe)] += 1

    total_par_mot = defaultdict(int)
    for (latin, _), nombre in couples.items():
        total_par_mot[latin] += nombre

    table = {}
    for (latin, arabe), nombre in sorted(couples.items(), key=lambda kv: -kv[1]):
        if latin in table or nombre < MIN_OBSERVATIONS:
            continue
        if nombre / total_par_mot[latin] >= PART_MINIMALE:
            table[latin] = arabe
    return table, len(couples)


def traduire(phrase, table):
    """Normalise une phrase arabizi. Retourne le texte et le nombre de mots inconnus.

    La table est consultee en premier ; les mots qu'elle ignore passent par les
    regles. Une table vide revient donc a n'utiliser que les regles.
    """
    sortie, inconnus = [], 0
    for mot in mots(phrase):
        cle = mot.lower()
        if cle in table:
            sortie.append(table[cle])
        else:
            sortie.append(par_regles(mot))
            inconnus += 1
    return " ".join(sortie), inconnus


if __name__ == "__main__":
    try:
        train = pd.read_csv(TRAIN)
        test = pd.read_csv(TEST)
    except FileNotFoundError:
        raise SystemExit(f"{TRAIN} ou {TEST} absent : lancer d'abord make_splits.py")

    table, observes = construire_table(train)

    predictions, inconnus_total, mots_total = [], 0, 0
    for phrase in test["transliteration"]:
        texte, inconnus = traduire(phrase, table)
        predictions.append(texte)
        inconnus_total += inconnus
        mots_total += len(mots(phrase))

    sortie = test[["id", "transliteration", "darija", "darija_norm",
                   "topic", "annotator_dialect", "mot_isole"]].copy()
    sortie["prediction"] = predictions
    sortie.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"{observes} couples de mots observes, {len(table)} retenus dans la table")
    print(f"{mots_total} mots a traduire, {inconnus_total} absents de la table "
          f"({100 * inconnus_total / mots_total:.1f}%) donc traites par regles")
    print(f"{len(sortie)} predictions enregistrees dans {OUTPUT}")
