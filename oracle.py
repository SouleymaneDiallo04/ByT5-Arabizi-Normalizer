"""Le modele ignore-t-il la bonne reponse, ou la classe-t-il deuxieme ?

La ventilation des erreurs attribue environ 37 % du cout restant a des mots que
le modele connait et qu'il choisit mal. Deux lectures s'opposent, et elles
n'appellent pas le meme remede :

  - le modele n'a pas la bonne reponse. Il lui manque de la capacite ou des
    donnees, et il faut un modele plus grand ou plus d'entrainement, soit une
    dizaine d'heures de calcul par essai ;
  - le modele a la bonne reponse mais la classe derriere une autre. Le probleme
    est alors le classement, et un reordonnanceur coute une fraction d'un
    entrainement.

Ce script tranche sans rien reentrainer. Il produit plusieurs hypotheses par
phrase, puis mesure ce que donnerait un choix parfait parmi elles. L'ecart
entre le premier choix et ce choix parfait est, par construction, tout ce
qu'un meilleur classement pourrait gagner : c'est une borne superieure, pas
une promesse.

Il ecrit deux fichiers au format attendu par evaluate.py et erreurs.py, afin
que la comparaison categorie par categorie dise non seulement combien le
classement pourrait gagner, mais sur quoi.

Usage :
    python oracle.py                       modele depuis Hugging Face
    python oracle.py chemin/vers/modele    modele local ou monte
"""

import sys
import time

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from evaluate import normaliser, distance, mesurer

MODELE = "sawti-nlp/byt5-arabizi-normalizer"
DEPOT_DONNEES = "sawti-nlp/darija-arabizi-corpus"
TEST = "test_real.csv"
SORTIE_PREMIER = "byt5_faisceau_premier.csv"
SORTIE_ORACLE = "byt5_faisceau_oracle.csv"

MAX_SOURCE, MAX_CIBLE = 192, 288
# Le lot est reduit : huit hypotheses par phrase font huit fois plus de
# sequences en vol, et la memoire d'un T4 n'est pas extensible.
TAILLE_LOT = 8
HYPOTHESES = 8

COLONNES = ["id", "transliteration", "darija", "darija_norm",
            "topic", "annotator_dialect", "mot_isole"]


def charger(chemin):
    try:
        return pd.read_csv(chemin)
    except FileNotFoundError:
        return pd.read_csv(hf_hub_download(DEPOT_DONNEES, f"splits/{chemin}",
                                           repo_type="dataset"))


def hypotheses(modele, tok, phrases, appareil, nombre=HYPOTHESES):
    """Rend, pour chaque phrase, la liste des hypotheses classees par le modele."""
    sorties = []
    for depart in range(0, len(phrases), TAILLE_LOT):
        lot = [str(p) for p in phrases[depart:depart + TAILLE_LOT]]
        entrees = tok(lot, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SOURCE).to(appareil)
        with torch.no_grad():
            g = modele.generate(**entrees, max_length=MAX_CIBLE,
                                num_beams=nombre, num_return_sequences=nombre)
        textes = tok.batch_decode(g, skip_special_tokens=True)
        for i in range(len(lot)):
            sorties.append(textes[i * nombre:(i + 1) * nombre])
    return sorties


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else MODELE
    if not torch.cuda.is_available():
        raise SystemExit("Ce diagnostic demande un GPU : il decode huit fois "
                         "plus qu'une inference ordinaire.")

    print(f"modele : {source}")
    tok = AutoTokenizer.from_pretrained(source)
    modele = AutoModelForSeq2SeqLM.from_pretrained(source).to("cuda").eval()
    test = charger(TEST)
    print(f"{len(test)} phrases, {HYPOTHESES} hypotheses chacune")

    debut = time.perf_counter()
    listes = hypotheses(modele, tok, test["transliteration"].tolist(), "cuda")
    print(f"decodage en {time.perf_counter() - debut:.0f} s")

    references = test["darija_norm"].map(normaliser).tolist()
    premiers, oracles, rangs, rang_exact = [], [], [], []
    for liste, reference in zip(listes, references):
        normalisees = [normaliser(h) for h in liste]
        premiers.append(liste[0])
        couts = [distance(h, reference) for h in normalisees]
        meilleur = min(range(len(couts)), key=couts.__getitem__)
        oracles.append(liste[meilleur])
        rangs.append(meilleur)
        rang_exact.append(next((i for i, h in enumerate(normalisees)
                                if h == reference), None))

    for nom, predictions in [(SORTIE_PREMIER, premiers), (SORTIE_ORACLE, oracles)]:
        sortie = test[COLONNES].copy()
        sortie["prediction"] = predictions
        sortie.to_csv(nom, index=False, encoding="utf-8-sig")

    premier = mesurer(references, [normaliser(p) for p in premiers])
    oracle = mesurer(references, [normaliser(p) for p in oracles])

    print(f"\n{'choix':<28} {'CER':>9} {'exactitude':>12}")
    print(f"{'premiere hypothese':<28} {premier['cer']:>9.4f} "
          f"{100 * premier['exactitude']:>11.1f}%")
    print(f"{'choix parfait parmi ' + str(HYPOTHESES):<28} {oracle['cer']:>9.4f} "
          f"{100 * oracle['exactitude']:>11.1f}%")
    print(f"{'ecart recuperable':<28} "
          f"{premier['cer'] - oracle['cer']:>9.4f} "
          f"{100 * (oracle['exactitude'] - premier['exactitude']):>+11.1f} pts")

    trouvees = [r for r in rang_exact if r is not None]
    print(f"\nreference exacte presente dans les {HYPOTHESES} hypotheses : "
          f"{len(trouvees)} phrases sur {len(test)} "
          f"({100 * len(trouvees) / len(test):.1f} %)")
    if trouvees:
        distribution = pd.Series(trouvees).value_counts().sort_index()
        print("  a quel rang : " +
              ", ".join(f"{r + 1} ({n})" for r, n in distribution.items()))
    print(f"rang moyen de la meilleure hypothese par la distance : "
          f"{sum(rangs) / len(rangs):.2f}")

    print(f"\nLe choix parfait n'est pas atteignable, il suppose de connaitre "
          f"la reponse.\nIl borne ce qu'un reordonnanceur pourrait gagner. Si "
          f"l'ecart est mince, le\nmodele ignore la reponse et c'est la "
          f"capacite qu'il faut augmenter.\n\n  python erreurs.py "
          f"{SORTIE_ORACLE} {SORTIE_PREMIER}\n\ndira sur quelles categories "
          f"ce gain potentiel se trouve.")
