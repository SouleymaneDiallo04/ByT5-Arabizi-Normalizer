"""Regles appliquees a l'arabizi avant le modele, contre ses defaillances mesurees.

La mesure de robustesse du 26 novembre (evaluation/robustesse_modele.py) a
releve trois defaillances qu'une regle sure peut traiter sans reentrainement :

  - repetitions : une lettre repetee plus de deux fois (bzaaaaf) fait parfois
    boucler le modele jusqu'a la longueur maximale. Elle est ramenee a deux :
    le doublement porte un sens en arabizi (mzyyan, bzzaf), pas l'allongement ;
  - emojis : un emoji fait parfois repeter un segment. Les emojis sont retires
    avant le modele et rendus a la fin de la sortie ;
  - phrases : sur une entree de plusieurs phrases, le modele s'arrete parfois
    apres le premier point. L'entree est coupee apres chaque . ! ou ? suivi
    d'une espace, chaque phrase est normalisee seule, et les sorties sont
    rejointes.

Chaque regle se mesure seule, sur le jeu de reglage d'abord. Le service
n'applique que celles nommees dans PRETRAITEMENT (service/config.py).
"""

import re

REGLES = ("repetitions", "emojis", "phrases")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
REPETITION = re.compile(r"([A-Za-z])\1{2,}", re.IGNORECASE)
FIN_DE_PHRASE = re.compile(r"(?<=[.!?])\s+")


def verifier(regles):
    inconnues = set(regles) - set(REGLES)
    if inconnues:
        raise ValueError(f"regles inconnues : {sorted(inconnues)} ; connues : {REGLES}")
    return tuple(regles)


def reduire_repetitions(texte):
    return REPETITION.sub(lambda m: m.group(0)[:2], texte)


def retirer_emojis(texte):
    """Le texte sans ses emojis, et les emojis dans l'ordre."""
    emojis = "".join(EMOJI.findall(texte))
    return re.sub(r"\s+", " ", EMOJI.sub("", texte)).strip(), emojis


def decouper(texte):
    return [p for p in FIN_DE_PHRASE.split(texte) if p]


def appliquer(normaliser, textes, regles):
    """Normalise des textes deja nettoyes, a travers les regles demandees.

    normaliser prend une liste de textes et rend, dans l'ordre, des triplets
    (sortie, confiance brute, complete), comme Normaliseur.normaliser. Une
    entree coupee en phrases prend la confiance la plus basse de ses phrases,
    et n'est complete que si toutes le sont.
    """
    regles = verifier(regles)
    if not regles:
        return normaliser(textes)
    preparees = []
    for t in textes:
        emojis = ""
        if "emojis" in regles:
            t, emojis = retirer_emojis(t)
        if "repetitions" in regles:
            t = reduire_repetitions(t)
        segments = decouper(t) if "phrases" in regles else ([t] if t else [])
        preparees.append((segments, emojis))

    a_decoder = [s for segments, _ in preparees for s in segments]
    resultats = iter(normaliser(a_decoder) if a_decoder else [])
    sorties = []
    for segments, emojis in preparees:
        parts = [next(resultats) for _ in segments]
        sortie = " ".join(s for s, _, _ in parts)
        if emojis:
            sortie = f"{sortie} {emojis}" if sortie else emojis
        sorties.append((sortie, min((c for _, c, _ in parts), default=1.0),
                        all(f for _, _, f in parts)))
    return sorties
