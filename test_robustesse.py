"""Controle de recette de l'exigence ENF-08 : resistance aux entrees bruitees.

Le cahier des charges demande qu'aucune exception non geree ne survienne sur cent
entrees bruitees, comportant fautes, allongements et ponctuation repetee. Les
entrees sont construites de facon deterministe, pour que deux executions
produisent le meme resultat (ENF-04).

Usage : python test_robustesse.py
"""

import pandas as pd

from baseline import construire_table, traduire

BASES = [
    "sbah lkhir", "wach nta mzyan", "3andek l7a9", "chrit 3 tomobilat",
    "mchit la pharmacie bach nakhod dwa", "salam 3likom", "kifach daba",
    "bslama 3likom", "achno kat3awdi", "had dwa mawaf9nich",
    "fin ghadi", "allah inej7k",
]
EMOJIS = "🙏😀🇲🇦"


def allonger(phrase):
    """Triple chaque voyelle : salam devient saaalaaam."""
    return "".join(c * 3 if c in "aeiou" else c for c in phrase)


def ponctuer(phrase):
    """Ajoute de la ponctuation repetee, comme dans les vrais messages."""
    return phrase + "???!!!..."


def fauter(phrase):
    """Echange deux lettres au milieu du mot le plus long, pour simuler une faute."""
    parts = phrase.split()
    if not parts:
        return phrase
        
    plus_long = max(range(len(parts)), key=lambda i: len(parts[i]))
    mot = parts[plus_long]
    if len(mot) > 2:
        milieu = len(mot) // 2
        mot = mot[:milieu - 1] + mot[milieu] + mot[milieu - 1] + mot[milieu + 1:]
    parts[plus_long] = mot
    return " ".join(parts)


def coller(phrase):
    """Supprime les espaces, cas frequent en messagerie."""
    return phrase.replace(" ", "")


def crier(phrase):
    """Met en majuscules, casse melangee incluse."""
    return phrase.upper()


def emojier(phrase):
    """Ajoute des emojis en fin de phrase."""
    return phrase + " " + EMOJIS


TRANSFORMATIONS = [allonger, ponctuer, fauter, coller, crier, emojier]

CAS_LIMITES = [
    "", " ", "   \t\n ", None, 0, 3, "3", "7", "9", "2",
    "ch", "kh", "ou", "é", "ç", "àéîôû", "ا", "سلام", "salam سلام",
    "???", "!!!", "...", "،،،", "@#$%^&*()", "---", "'''",
    "123456789", "3 7 9", "a" * 500, "sa" * 200, "\n\t", "NULL", "nan",
    "mrc bzf nchlh", "l'3adres dyal l'ecole", "3la slamtk", "7amdoulah 😀",
]


def entrees_bruitees():
    """Construit la liste deterministe des entrees du controle."""
    cas = [transformer(base) for base in BASES for transformer in TRANSFORMATIONS]
    return cas + CAS_LIMITES


if __name__ == "__main__":
    entrees = entrees_bruitees()
    print(f"{len(entrees)} entrees bruitees construites "
          f"({len(BASES)} bases x {len(TRANSFORMATIONS)} transformations "
          f"+ {len(CAS_LIMITES)} cas limites)")

    tables = {"regles seules": {}}
    try:
        tables["regles + table"] = construire_table(pd.read_csv("train.csv"))[0]
    except FileNotFoundError:
        print("train.csv absent : seules les regles sont controlees")

    echecs = 0
    for nom, table in tables.items():
        rates = []
        for entree in entrees:
            try:
                texte, _ = traduire(entree, table)
                if not isinstance(texte, str):
                    rates.append((entree, "sortie non textuelle"))
            except Exception as erreur:
                rates.append((entree, f"{type(erreur).__name__}: {erreur}"))
        echecs += len(rates)
        etat = "PASSE" if not rates else "ECHEC"
        print(f"  [{etat}] {nom:18} {len(entrees) - len(rates)}/{len(entrees)} "
              f"traitees sans exception")
        for entree, cause in rates[:5]:
            print(f"      {entree!r:40} {cause}")

    assert len(entrees) >= 100, f"ENF-08 exige cent entrees, {len(entrees)} fournies"
    raise SystemExit(0 if echecs == 0 else 1)
