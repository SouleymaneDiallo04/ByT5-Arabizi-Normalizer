"""Profil d'un corpus candidat, mesure avant de depenser un entrainement.

Ajouter des donnees coute entre sept et quinze heures de GPU par essai. La
question n'est pas de savoir si un corpus est gros, mais s'il apporte ce qui
manque. Ce script repond sans entrainer.

Trois mesures, par ordre d'importance.

**La couverture du vocabulaire manquant.** L'analyse d'erreurs a etabli que
22,9 % des mots faux du modele sont des formes **jamais vues** dans le corpus
d'entrainement, concentrees sur Guelmim a 158,7 pour mille et Tetouan a 108,4
contre 19,1 a Marrakech. Un corpus candidat qui ne couvre pas ces formes
apportera du volume sur ce que le modele reussit deja. C'est le chiffre qui
decide.

**La distance de profil orthographique au jeu de test.** Les sources existantes
n'ecrivent pas pareil : il avait ete mesure que daa-pairs est la plus proche de
TerjamaBench et DODa-10K la plus lointaine. Le profil est redefini ici
explicitement et recalcule pour toutes les sources, de sorte que les valeurs
soient comparables entre elles. Un corpus qui ecrit autrement que le jeu de
test enseigne une convention qui sera comptee comme erreur.

**La distribution des longueurs.** Un corpus de paragraphes ne prepare pas aux
phrases courtes du jeu de test, dont 163 sur 850 sont des mots isoles.

Usage :
    python profil_corpus.py nilechat_translit.csv
    python profil_corpus.py un.csv deux.csv      plusieurs candidats
"""

import collections
import re
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

from baseline import mots
from evaluate import normaliser

DEPOT = "sawti-nlp/darija-arabizi-corpus"
TRAIN, TEST = "train.csv", "test_real.csv"
CORPUS = "darija_all.csv"

VOCALISATION = re.compile("[ً-ٕـٰ]")
HAMZAS = re.compile("[أإآءئؤ]")
CHIFFRES_LETTRES = re.compile("[23579]")
LATIN = re.compile("[A-Za-z]")


def charger(chemin, colonnes=None):
    try:
        return pd.read_csv(chemin, usecols=colonnes)
    except FileNotFoundError:
        return pd.read_csv(hf_hub_download(DEPOT, f"splits/{chemin}",
                                           repo_type="dataset"),
                           usecols=colonnes)


def profil(arabe, latin):
    """Vecteur de traits orthographiques, chacun une part entre 0 et 1.

    Les traits sont choisis parce qu'ils distinguent reellement les sources du
    projet, et non pour leur interet linguistique : vocalisation, hamzas, ta
    marbuta, ponctuation arabe cote cible ; chiffres-lettres et conventions ch
    ou sh cote arabizi.
    """
    ma = [m for t in arabe for m in mots(normaliser(str(t)))]
    textes_a = [str(t) for t in arabe]
    textes_l = [str(t).lower() for t in latin]
    n = max(len(ma), 1)
    na = max(len(textes_a), 1)
    joint = " ".join(textes_l)
    return {
        "vocalisation": sum(1 for m in ma if VOCALISATION.search(m)) / n,
        "hamzas": sum(1 for m in ma if HAMZAS.search(m)) / n,
        "ta marbuta": sum(1 for m in ma if m.endswith("ة")) / n,
        "alif maqsura": sum(1 for m in ma if m.endswith("ى")) / n,
        "point d_interrogation arabe":
            sum(1 for t in textes_a if "؟" in t) / na,
        "virgule arabe": sum(1 for t in textes_a if "،" in t) / na,
        "latin dans la cible": sum(1 for t in textes_a if LATIN.search(t)) / na,
        "chiffres-lettres":
            sum(1 for t in textes_l if CHIFFRES_LETTRES.search(t)) / na,
        "ch contre sh": joint.count("ch") / max(joint.count("ch") + joint.count("sh"), 1),
        "ou contre u": joint.count("ou") / max(joint.count("ou") + joint.count("u"), 1),
    }


def distance(a, b):
    """Somme des ecarts absolus entre deux profils. Zero vaut identite."""
    return sum(abs(a[k] - b[k]) for k in a)


def octets(serie):
    tailles = sorted(len(str(t).encode()) for t in serie)
    n = len(tailles)
    return (tailles[n // 2], tailles[int(n * 0.9)]) if n else (0, 0)


if __name__ == "__main__":
    candidats = sys.argv[1:]
    if not candidats:
        raise SystemExit("Usage : python profil_corpus.py un_corpus.csv")

    test = charger(TEST, ["darija_norm", "transliteration"])
    train = charger(TRAIN, ["darija", "transliteration", "source"])

    # Les formes que le jeu de test attend et que le corpus actuel n'a jamais
    # montrees. C'est la lacune a combler, et elle est ponderee par le nombre
    # d'occurrences : une forme attendue cinq fois pese cinq fois.
    vu = set()
    for t in train["darija"]:
        vu.update(mots(normaliser(str(t))))
    attendu = collections.Counter()
    for t in test["darija_norm"]:
        attendu.update(mots(normaliser(str(t))))
    manquantes = {m: n for m, n in attendu.items() if m not in vu}
    poids = sum(manquantes.values())
    print(f"vocabulaire du jeu de test : {len(attendu)} formes distinctes")
    print(f"  absentes du corpus d'entrainement : {len(manquantes)} formes, "
          f"{poids} occurrences")

    reference = profil(test["darija_norm"], test["transliteration"])

    jeux = [(c, charger(c, ["darija", "transliteration"])) for c in candidats]
    # Les sources deja employees servent de temoin. Elles doivent sortir a
    # 0 % de couverture, puisqu'elles constituent le corpus d'entrainement et
    # que la lacune est definie comme ce qui n'y figure pas. Un candidat qui
    # depasse nettement ce plancher apporte du vocabulaire reellement neuf.
    # darija_all.csv est prefere quand il est la, car il precede le decoupage ;
    # a defaut train.csv porte la meme colonne source et suffit.
    try:
        tout = pd.read_csv(CORPUS, usecols=["darija", "transliteration", "source"])
    except (FileNotFoundError, ValueError):
        tout = train
    for nom, part in tout.groupby("source"):
        jeux.append((f"[temoin] {nom}", part))

    print(f"\n{'corpus':<34} {'paires':>9} {'formes':>8} {'manquantes':>11} "
          f"{'distance':>9} {'octets med':>11} {'p90':>6}")
    for nom, d in jeux:
        formes = set()
        for t in d["darija"]:
            formes.update(mots(normaliser(str(t))))
        couvertes = sum(n for m, n in manquantes.items() if m in formes)
        echantillon = d if len(d) <= 40000 else d.sample(40000, random_state=42)
        p = profil(echantillon["darija"], echantillon["transliteration"])
        med, p90 = octets(echantillon["transliteration"])
        print(f"{nom[:34]:<34} {len(d):>9} {len(formes):>8} "
              f"{100 * couvertes / max(poids, 1):>10.1f}% "
              f"{distance(reference, p):>9.3f} {med:>11} {p90:>6}")

    # Une distance globale ne dit pas si l'ecart est inoffensif. Un corpus qui
    # ponctue autrement se normalise, un corpus qui n'ecrit pas les memes mots
    # ne se rattrape pas. Le detail n'est rendu que pour les candidats : les
    # temoins alourdiraient sans rien apprendre.
    for nom, d in jeux[:len(candidats)]:
        echantillon = d if len(d) <= 40000 else d.sample(40000, random_state=42)
        p = profil(echantillon["darija"], echantillon["transliteration"])
        print(f"\ndetail du profil : {nom}")
        print(f"  {'trait':<30} {'candidat':>10} {'jeu de test':>13} {'ecart':>9}")
        for trait, valeur in sorted(p.items(),
                                    key=lambda kv: -abs(kv[1] - reference[kv[0]])):
            print(f"  {trait:<30} {valeur:>10.3f} {reference[trait]:>13.3f} "
                  f"{valeur - reference[trait]:>+9.3f}")

    print(f"\n  La colonne manquantes donne la part des occurrences absentes du"
          f"\n  corpus actuel que chaque source couvrirait. C'est elle qui dit si"
          f"\n  un candidat comble la lacune mesuree ou ajoute du volume la ou le"
          f"\n  modele reussit deja.")
    print(f"\n  La distance compare le profil orthographique au jeu de test, sur"
          f"\n  dix traits. Plus elle est basse, plus la source ecrit comme ce sur"
          f"\n  quoi le modele sera note.")
