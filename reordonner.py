"""Reordonner les hypotheses du faisceau, sans reentrainer le modele.

Le diagnostic de oracle.py a etabli le fait central : la reference exacte est
presente dans les huit hypotheses pour 54,2 % des phrases, et le modele n'en
choisit que 30,1 %. Il a la bonne reponse sous la main dans 205 cas sur 850 et
classe autre chose devant.

Le signe le plus parlant est que chercher davantage degrade le premier choix :
le decodage glouton rend un CER de 0,1093, huit faisceaux 0,1161. La fonction
que le modele maximise est la vraisemblance de la sequence, et elle n'est pas
le CER. Le probleme n'est donc ni la capacite ni les donnees, c'est le critere
de selection.

Trois selections sont comparees, de la plus simple a la plus reglee.

  Premiere hypothese   ce que rend le faisceau, point de reference.
  Consensus            choisir l'hypothese la plus proche de toutes les autres,
                       ponderee par leur probabilite. C'est le choix qui
                       minimise la distance d'edition attendue, donc qui vise
                       la metrique au lieu de la vraisemblance. Aucun reglage.
  Melange regle        combinaison du score du modele, du consensus et d'un
                       modele de langue sur les cibles d'entrainement.

Les poids sont regles deux fois, et c'est leur comparaison qui est
l'experience. Une fois sur le jeu de developpement, qui vient des sources
d'entrainement et ou le modele atteint 0,0473 de CER ; une fois sur un
prelevement du jeu de test, ou il est a 0,1087. Un premier essai, regle sur le
seul developpement, avait echoue : les poids appris sur des faisceaux ou le
premier choix est presque toujours bon ne transferaient pas.

Un ecart entre les deux lignes accuse le jeu de reglage. Leur egalite accuse
les signaux eux-memes, et ferme la piste.

Dans les deux cas l'evaluation porte sur les 678 phrases qui n'ont servi a
aucun reglage.

Le plafond, soit le choix parfait parmi les huit, vaut 0,0635 de CER. Il est
inatteignable puisqu'il suppose de connaitre la reponse ; il borne ce que
n'importe quelle methode de cette famille peut esperer.

Usage :
    python reordonner.py                       modele depuis Hugging Face
    python reordonner.py chemin/vers/modele    modele local ou monte
"""

import collections
import itertools
import math
import sys
import time

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from baseline import mots
from decouper_test import prelever
from evaluate import normaliser, distance, mesurer

MODELE = "sawti-nlp/byt5-arabizi-normalizer"
DEPOT_DONNEES = "sawti-nlp/darija-arabizi-corpus"
TRAIN, DEV, TEST = "train.csv", "dev.csv", "test_real.csv"
SORTIE = "byt5_reordonne.csv"
REFERENCE = "byt5_predictions.csv"
REFERENCE_RESTREINTE = "byt5_predictions_evaluation.csv"

MAX_SOURCE, MAX_CIBLE = 192, 288
TAILLE_LOT = 8
HYPOTHESES = 8
ECHANTILLON_DEV = 1500
LISSAGE = 0.5
PAS_GRILLE = 0.1
GRAINE = 42

COLONNES = ["id", "transliteration", "darija", "darija_norm",
            "topic", "annotator_dialect", "mot_isole"]


def charger(chemin, echantillon=None):
    try:
        d = pd.read_csv(chemin)
    except FileNotFoundError:
        d = pd.read_csv(hf_hub_download(DEPOT_DONNEES, f"splits/{chemin}",
                                        repo_type="dataset"))
    if echantillon and echantillon < len(d):
        d = d.sample(echantillon, random_state=GRAINE).reset_index(drop=True)
    return d


def hypotheses(modele, tok, phrases, appareil):
    """Rend, par phrase, les hypotheses du faisceau et leur score de sequence."""
    textes, scores = [], []
    for depart in range(0, len(phrases), TAILLE_LOT):
        lot = [str(p) for p in phrases[depart:depart + TAILLE_LOT]]
        entrees = tok(lot, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SOURCE).to(appareil)
        with torch.no_grad():
            g = modele.generate(**entrees, max_length=MAX_CIBLE,
                                num_beams=HYPOTHESES,
                                num_return_sequences=HYPOTHESES,
                                output_scores=True, return_dict_in_generate=True)
        decodes = tok.batch_decode(g.sequences, skip_special_tokens=True)
        bruts = g.sequences_scores.tolist()
        for i in range(len(lot)):
            tranche = slice(i * HYPOTHESES, (i + 1) * HYPOTHESES)
            textes.append([normaliser(h) for h in decodes[tranche]])
            scores.append(bruts[tranche])
    return textes, scores


def modele_de_langue(cibles):
    """Bigrammes de mots sur les cibles d'entrainement, lissage additif.

    La vraisemblance de ByT5 se calcule octet par octet. Un modele de mots
    apporte un avis sur la fluidite en ecriture arabe que le premier ne voit
    qu'indirectement, et c'est cette independance qui le rend utile ici.
    """
    unigrammes, bigrammes = collections.Counter(), collections.Counter()
    for cible in cibles:
        suite = ["<d>"] + mots(normaliser(str(cible))) + ["<f>"]
        unigrammes.update(suite)
        bigrammes.update(zip(suite, suite[1:]))
    return unigrammes, bigrammes, len(unigrammes)


def score_langue(texte, unigrammes, bigrammes, vocabulaire):
    suite = ["<d>"] + mots(texte) + ["<f>"]
    if len(suite) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(suite, suite[1:]):
        total += math.log((bigrammes[(a, b)] + LISSAGE)
                          / (unigrammes[a] + LISSAGE * vocabulaire))
    return total / (len(suite) - 1)


def consensus(liste, scores):
    """Distance d'edition attendue de chaque hypothese aux autres.

    Ponderee par la probabilite que le modele accorde a chacune : une hypothese
    proche de celles que le modele juge vraisemblables est un meilleur pari
    qu'une hypothese proche de ses hypotheses marginales.
    """
    poids = [math.exp(s) for s in scores]
    somme = sum(poids) or 1.0
    poids = [p / somme for p in poids]
    return [sum(poids[j] * distance(liste[i], liste[j])
                for j in range(len(liste)) if j != i)
            for i in range(len(liste))]


def centrer(valeurs):
    """Ramene une serie dans [0,1] a l'interieur d'une phrase.

    Les trois signaux n'ont ni la meme echelle ni la meme unite. Les comparer
    phrase par phrase, et non globalement, evite qu'une phrase longue pese plus
    qu'une courte dans le reglage des poids.
    """
    bas, haut = min(valeurs), max(valeurs)
    if haut - bas < 1e-12:
        return [0.5] * len(valeurs)
    return [(v - bas) / (haut - bas) for v in valeurs]


def traits(listes, scores, langue):
    """Trois signaux par hypothese, chacun ramene dans [0,1], plus haut est mieux."""
    unigrammes, bigrammes, vocabulaire = langue
    sortie = []
    for liste, score in zip(listes, scores):
        modele = centrer(score)
        accord = centrer([-c for c in consensus(liste, score)])
        fluidite = centrer([score_langue(h, unigrammes, bigrammes, vocabulaire)
                            for h in liste])
        sortie.append(list(zip(modele, accord, fluidite)))
    return sortie


def choisir(traits_phrase, poids):
    melange = [sum(p * t for p, t in zip(poids, trio)) for trio in traits_phrase]
    return max(range(len(melange)), key=melange.__getitem__)


def cer(listes, choix, references):
    erreurs = sum(distance(liste[i], r) for liste, i, r in zip(listes, choix, references))
    longueurs = sum(len(r) for r in references) or 1
    exactes = sum(liste[i] == r for liste, i, r in zip(listes, choix, references))
    return erreurs / longueurs, exactes / max(len(references), 1)


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else MODELE
    if not torch.cuda.is_available():
        raise SystemExit("Ce script demande un GPU : il decode huit hypotheses "
                         "par phrase, sur le dev et sur le test.")

    print(f"modele : {source}")
    tok = AutoTokenizer.from_pretrained(source)
    modele = AutoModelForSeq2SeqLM.from_pretrained(source).to("cuda").eval()

    print("modele de langue sur les cibles d'entrainement...")
    langue = modele_de_langue(charger(TRAIN)["darija_norm"])
    print(f"  {langue[2]} formes, {len(langue[1])} bigrammes")

    # Le jeu de test est decode une seule fois, puis partage entre reglage et
    # evaluation : decoder deux fois les memes phrases couterait le double sans
    # rien apporter.
    test = charger(TEST)
    reglage, evaluation = prelever(test)
    debut = time.perf_counter()
    listes_t, scores_t = hypotheses(modele, tok, test["transliteration"].tolist(), "cuda")
    print(f"test : {len(test)} phrases decodees en {time.perf_counter() - debut:.0f} s")
    traits_t = traits(listes_t, scores_t, langue)
    refs_t = test["darija_norm"].map(normaliser).tolist()
    rang = {ident: i for i, ident in enumerate(test["id"])}
    i_reglage = [rang[x] for x in reglage["id"]]
    i_evaluation = [rang[x] for x in evaluation["id"]]

    dev = charger(DEV, ECHANTILLON_DEV)
    debut = time.perf_counter()
    listes_d, scores_d = hypotheses(modele, tok, dev["transliteration"].tolist(), "cuda")
    print(f"dev  : {len(dev)} phrases decodees en {time.perf_counter() - debut:.0f} s")
    traits_d = traits(listes_d, scores_d, langue)
    refs_d = dev["darija_norm"].map(normaliser).tolist()

    pas = int(round(1 / PAS_GRILLE))
    grille = [(a / pas, b / pas, (pas - a - b) / pas)
              for a in range(pas + 1) for b in range(pas + 1 - a)]

    def regler(listes, traits_, references, indices=None):
        """Cherche les poids qui minimisent le CER sur le jeu fourni."""
        if indices is not None:
            listes = [listes[i] for i in indices]
            traits_ = [traits_[i] for i in indices]
            references = [references[i] for i in indices]
        meilleur, retenus = None, None
        for poids in grille:
            valeur = cer(listes, [choisir(x, poids) for x in traits_], references)[0]
            if meilleur is None or valeur < meilleur:
                meilleur, retenus = valeur, poids
        return retenus, meilleur

    # Deux reglages, et c'est leur comparaison qui est l'experience. Le premier
    # sur le jeu de developpement, qui vient des sources d'entrainement et ou le
    # modele reussit deux fois et demie mieux ; le second sur un prelevement du
    # jeu de test, donc sur la distribution ou le probleme existe reellement.
    p_dev, c_dev = regler(listes_d, traits_d, refs_d)
    p_reg, c_reg = regler(listes_t, traits_t, refs_t, i_reglage)
    print(f"\n{'reglage':<22} {'modele':>7} {'consensus':>10} {'langue':>7} "
          f"{'CER sur son jeu':>17}")
    for nom, p, c in [("developpement", p_dev, c_dev), ("prelevement du test", p_reg, c_reg)]:
        print(f"{nom:<22} {p[0]:>7.1f} {p[1]:>10.1f} {p[2]:>7.1f} {c:>17.4f}")

    sous = lambda s: [s[i] for i in i_evaluation]
    listes_e, traits_e, refs_e = sous(listes_t), sous(traits_t), sous(refs_t)
    choix_reg = [choisir(x, p_reg) for x in traits_e]

    lignes = [
        ("premiere hypothese", [0] * len(listes_e)),
        ("consensus seul, sans reglage", [choisir(x, (0, 1, 0)) for x in traits_e]),
        ("melange regle sur le dev", [choisir(x, p_dev) for x in traits_e]),
        ("melange regle sur le test", choix_reg),
        ("choix parfait (plafond)",
         [min(range(len(l)), key=lambda i: distance(l[i], r))
          for l, r in zip(listes_e, refs_e)]),
    ]
    print(f"\nmesure sur les {len(listes_e)} phrases d'evaluation")
    print(f"{'selection':<32} {'CER':>9} {'exactitude':>12}")
    for nom, choix in lignes:
        c, e = cer(listes_e, choix, refs_e)
        print(f"{nom:<32} {c:>9.4f} {100 * e:>11.1f}%")

    print(f"\n  Comparer les deux lignes de melange repond a une seule question :"
          f"\n  l'echec du reordonnancement venait-il des signaux ou du jeu sur "
          f"lequel\n  on les a pondere ? Un ecart entre elles accuse le jeu, "
          f"leur egalite\n  accuse les signaux.")
    print(f"\n  Prudence : les poids du test sont regles sur {len(i_reglage)} "
          f"phrases seulement.\n  Comparer le CER de reglage ci-dessus a celui "
          f"d'evaluation dit si le\n  reglage a capte du signal ou du bruit.")

    sortie = evaluation[COLONNES].copy()
    sortie["prediction"] = [liste[i] for liste, i in zip(listes_e, choix_reg)]
    sortie.to_csv(SORTIE, index=False, encoding="utf-8-sig")
    print(f"\n{SORTIE} ecrit, {len(sortie)} phrases.")

    # La reference du run glouton porte sur les 850 phrases. La ventiler contre
    # 678 comparerait deux populations differentes et rendrait des ecarts qui
    # n'existent pas : une copie restreinte aux memes phrases est donc ecrite.
    try:
        reference = pd.read_csv(REFERENCE)
    except FileNotFoundError:
        print(f"  {REFERENCE} absent : deposer le fichier pour pouvoir comparer.")
    else:
        restreinte = reference[reference["id"].isin(set(evaluation["id"]))]
        restreinte.to_csv(REFERENCE_RESTREINTE, index=False, encoding="utf-8-sig")
        print(f"  {REFERENCE_RESTREINTE} ecrit, {len(restreinte)} phrases, "
              f"pour comparer a population egale.")
        print(f"\n  python erreurs.py {SORTIE} {REFERENCE_RESTREINTE}")
