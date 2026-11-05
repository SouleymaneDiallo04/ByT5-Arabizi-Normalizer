"""Mesures d'inference sur le modele affine.

Quatre exigences du cahier des charges se mesurent sur le modele entraine, sans
le reentrainer, et tiennent donc dans une seule session :

  PO-06   decodage glouton contre recherche en faisceaux
  EF-03   score de confiance dans [0,1], calibre sur le jeu de developpement
  ENF-02  cout de la quantification en entier 8 bits
  ENF-03  latence de decodage d'une phrase seule

Les mesures processeur sont volontairement appariees sur le meme sous-ensemble :
ce qui doit etre decide est l'ecart entre fp32 et int8, pas leur valeur absolue,
et un sous-ensemble appariee suffit a l'etablir pour un cout bien moindre.

Dependances : requirements-train.txt.

Usage :
    python mesures.py                      modele depuis Hugging Face
    python mesures.py chemin/vers/modele   modele local ou monte
"""

import io
import sys
import time

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from evaluate import normaliser, mesurer

MODELE = "sawti-nlp/byt5-arabizi-normalizer"
DEPOT_DONNEES = "sawti-nlp/darija-arabizi-corpus"
DEV, TEST = "dev.csv", "test_real.csv"

MAX_SOURCE, MAX_CIBLE = 192, 288
TAILLE_LOT = 16
FAISCEAUX = (1, 4)

ECHANTILLON_CPU = 200       # phrases pour les variantes processeur, appariees
ECHANTILLON_DEV = 2000      # phrases pour calibrer le score de confiance
ECHANTILLON_LATENCE = 100   # phrases chronometrees une par une
DECILES = 10
GRAINE = 42

COLONNES = ["id", "transliteration", "darija", "darija_norm",
            "topic", "annotator_dialect", "mot_isole"]


def charger(chemin, echantillon=None):
    """Lit un fichier de paires, localement puis a defaut depuis Hugging Face."""
    try:
        d = pd.read_csv(chemin)
    except FileNotFoundError:
        d = pd.read_csv(hf_hub_download(DEPOT_DONNEES, f"splits/{chemin}",
                                        repo_type="dataset"))
    if echantillon and echantillon < len(d):
        d = d.sample(echantillon, random_state=GRAINE).reset_index(drop=True)
    return d


def predire(modele, tok, phrases, appareil, faisceaux=1, confiance=False):
    """Decode une liste de phrases, et renvoie eventuellement la confiance.

    La confiance retenue est la probabilite moyenne par octet produit, soit
    l'exponentielle de la moyenne des log-probabilites de transition. Elle vit
    dans [0,1] par construction, ce qu'exige EF-03, elle ne demande aucun
    reentrainement, et elle se compare d'une phrase a l'autre quelle que soit
    leur longueur, ce qu'une somme de log-probabilites ne permettrait pas.
    """
    textes, confiances = [], []
    for depart in range(0, len(phrases), TAILLE_LOT):
        lot = [str(p) for p in phrases[depart:depart + TAILLE_LOT]]
        entrees = tok(lot, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SOURCE).to(appareil)
        with torch.no_grad():
            sortie = modele.generate(**entrees, max_length=MAX_CIBLE,
                                     num_beams=faisceaux,
                                     output_scores=confiance,
                                     return_dict_in_generate=True)
        textes.extend(tok.batch_decode(sortie.sequences, skip_special_tokens=True))

        if confiance:
            transitions = modele.compute_transition_scores(
                sortie.sequences, sortie.scores, normalize_logits=True)
            # generate complete apres la fin de sequence avec le jeton de
            # remplissage : le compter gonflerait la confiance, le modele lui
            # donnant une probabilite proche de 1. Chez ByT5 seul le
            # remplissage porte l'identifiant 0, les octets commencent a 3.
            jetons = sortie.sequences[:, 1:]
            garde = jetons != tok.pad_token_id
            transitions = transitions.masked_fill(~garde, 0.0)
            moyenne = transitions.sum(-1) / garde.sum(-1).clamp(min=1)
            confiances.extend(moyenne.exp().tolist())

    return textes, (confiances if confiance else None)


def latence(modele, tok, phrases, appareil, lot=1):
    """Chronometre le decodage, phrase par phrase ou par lot.

    ENF-03 demande les deux : une phrase seule est le cas d'usage de l'API
    interactive, le lot celui de EF-06. Le premier appel paie l'allocation des
    noyaux de calcul, deux groupes de chauffe sont donc jetes avant de mesurer.
    """
    groupes = [phrases[i:i + lot] for i in range(0, len(phrases), lot)]

    def passe(groupe):
        entrees = tok([str(p) for p in groupe], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=MAX_SOURCE).to(appareil)
        with torch.no_grad():
            modele.generate(**entrees, max_length=MAX_CIBLE)

    for groupe in groupes[:2]:
        passe(groupe)

    temps = []
    for groupe in groupes:
        if appareil == "cuda":
            torch.cuda.synchronize()
        debut = time.perf_counter()
        passe(groupe)
        if appareil == "cuda":
            torch.cuda.synchronize()
        temps.append(time.perf_counter() - debut)

    s = pd.Series(temps)
    return {"mediane": s.median(), "p90": s.quantile(0.9),
            "par_phrase": s.sum() / max(len(phrases), 1), "groupes": len(groupes)}


def taille_mo(modele):
    """Poids du modele serialise, en megaoctets."""
    tampon = io.BytesIO()
    torch.save(modele.state_dict(), tampon)
    return tampon.tell() / 1e6


def quantifier(modele):
    """Quantification dynamique des couches lineaires en entier 8 bits.

    C'est la seule qui ne demande aucune dependance supplementaire et qui vise
    le processeur, ou le service tournera une fois conteneurise.
    """
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(modele, {torch.nn.Linear}, dtype=torch.qint8)


def table_confiance(reference, prediction, confiance, libelle):
    """Affiche la justesse par decile de confiance, et la correlation de rang.

    Un score de confiance n'a de valeur que s'il est monotone : une confiance
    plus haute doit annoncer une erreur plus basse. La correlation de Spearman
    resume cette monotonie en un chiffre.
    """
    t = pd.DataFrame({"reference": list(reference), "prediction": list(prediction),
                      "confiance": list(confiance)})
    juste = t["prediction"].eq(t["reference"]).astype(float)
    rho = t["confiance"].corr(juste, method="spearman")

    print(f"\n  {libelle} : {len(t)} phrases, correlation de Spearman "
          f"confiance / justesse {rho:+.3f}")
    print(f"  {'decile':>7} {'phrases':>8} {'confiance':>10} {'CER':>8} "
          f"{'exactitude':>12}")
    t["decile"] = pd.qcut(t["confiance"], DECILES, labels=False,
                          duplicates="drop")
    # Des confiances toutes egales ne produisent aucune borne : qcut rend alors
    # des valeurs absentes et le tableau sortirait vide sans un mot.
    if t["decile"].isna().all():
        t["decile"] = 0
    for decile, part in t.groupby("decile"):
        m = mesurer(part["reference"], part["prediction"])
        print(f"  {int(decile) + 1:>7} {len(part):>8} "
              f"{part['confiance'].mean():>10.3f} {m['cer']:>8.4f} "
              f"{100 * m['exactitude']:>11.1f}%")
    return t


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else MODELE
    print(f"modele  : {source}")

    tok = AutoTokenizer.from_pretrained(source)
    test = charger(TEST)
    print(f"jeu de test : {len(test)} phrases")

    # Sous-ensemble des mesures lourdes. Il est tire au hasard et non pris en
    # tete : le jeu de test est ordonne par theme, et ses premieres lignes sont
    # celles des themes les plus difficiles, ce qui fausserait les valeurs
    # absolues. La graine le rend reproductible, comme l'exige ENF-04.
    echantillon = test.sample(ECHANTILLON_CPU, random_state=GRAINE)
    phrases_e = echantillon["transliteration"].tolist()
    reference_e = echantillon["darija_norm"].map(normaliser)
    latences = []   # (variante, appareil, phrase seule, lot) pour ENF-03

    if torch.cuda.is_available():
        modele = AutoModelForSeq2SeqLM.from_pretrained(source).to("cuda").eval()
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

        print("\n== PO-06  strategie de decodage "
              "=========================================")
        print(f"{'faisceaux':>10} {'CER':>9} {'exactitude':>12} {'secondes':>10}")
        for n in FAISCEAUX:
            debut = time.perf_counter()
            pred, _ = predire(modele, tok, test["transliteration"].tolist(),
                              "cuda", faisceaux=n)
            duree = time.perf_counter() - debut
            m = mesurer(test["darija_norm"].map(normaliser),
                        [normaliser(p) for p in pred])
            print(f"{n:>10} {m['cer']:>9.4f} {100 * m['exactitude']:>11.1f}% "
                  f"{duree:>10.1f}")
            sortie = test[COLONNES].copy()
            sortie["prediction"] = pred
            sortie.to_csv(f"byt5_predictions_faisceaux{n}.csv", index=False,
                          encoding="utf-8-sig")
        print("\n  ventilation complete : python evaluate.py "
              "byt5_predictions_faisceaux4.csv")

        print("\n== EF-03  score de confiance "
              "=============================================")
        # Le calcul de confiance s'appuie sur compute_transition_scores, dont la
        # signature a bouge d'une version de transformers a l'autre. Un echec ici
        # ne doit pas emporter les sections suivantes, qui coutent une heure.
        try:
            dev = charger(DEV, ECHANTILLON_DEV)
            for libelle, d in [("developpement", dev), ("test", test)]:
                pred, conf = predire(modele, tok, d["transliteration"].tolist(),
                                     "cuda", faisceaux=1, confiance=True)
                t = table_confiance(d["darija_norm"].map(normaliser),
                                    [normaliser(p) for p in pred], conf, libelle)
                t.to_csv(f"confiance_{libelle}.csv", index=False,
                         encoding="utf-8-sig")
        except Exception as erreur:
            import traceback
            traceback.print_exc()
            print(f"\n  EF-03 n'a pas abouti ({type(erreur).__name__}). "
                  "Les mesures suivantes continuent.")
        print("\n  Le dev calibre le score, le test verifie qu'il se transfere "
              "a des\n  phrases d'une autre origine. Une monotonie qui ne "
              "survivrait pas au\n  transfert serait un surajustement du seuil, "
              "pas une mesure de confiance.")

        latences.append(("fp32", "GPU",
                         latence(modele, tok, phrases_e[:ECHANTILLON_LATENCE],
                                 "cuda", 1),
                         latence(modele, tok, phrases_e, "cuda", TAILLE_LOT)))

        del modele
        torch.cuda.empty_cache()
    else:
        print("\nAucun GPU : PO-06 et EF-03 sont sautes, ils demandent un "
              "decodage repete.")

    print("\n== ENF-02  cout de la quantification "
          "=====================================")
    print(f"  qualite sur {len(echantillon)} phrases tirees au hasard, "
          f"les memes pour les deux variantes")
    print(f"\n{'variante':>10} {'taille Mo':>11} {'CER':>9} {'exactitude':>12}")

    for nom in ("fp32", "int8"):
        m = AutoModelForSeq2SeqLM.from_pretrained(source).eval()
        if nom == "int8":
            m = quantifier(m)
        pred, _ = predire(m, tok, phrases_e, "cpu")
        q = mesurer(reference_e, [normaliser(p) for p in pred])
        print(f"{nom:>10} {taille_mo(m):>11.0f} {q['cer']:>9.4f} "
              f"{100 * q['exactitude']:>11.1f}%")
        latences.append((nom, "CPU",
                         latence(m, tok, phrases_e[:ECHANTILLON_LATENCE],
                                 "cpu", 1),
                         latence(m, tok, phrases_e, "cpu", TAILLE_LOT)))
        del m

    print("\n== ENF-03  latence, pour une phrase et pour un lot "
          "=======================")
    print(f"{'variante':>9} {'appareil':>9} | {'phrase med':>11} {'p90':>9} | "
          f"{'lot de ' + str(TAILLE_LOT):>12} {'p90':>9} {'par phrase':>11}")
    for nom, appareil, seule, groupe in latences:
        print(f"{nom:>9} {appareil:>9} | {seule['mediane']:>10.3f}s "
              f"{seule['p90']:>8.3f}s | {groupe['mediane']:>11.3f}s "
              f"{groupe['p90']:>8.3f}s {groupe['par_phrase']:>10.3f}s")

    print("\n  La mediane est le ressenti courant, le p90 ce que subit un "
          "utilisateur\n  sur dix : c'est lui qui doit tenir la cible. La "
          "colonne par phrase mesure\n  ce que coute une phrase une fois "
          "amortie dans un lot, soit le regime de\n  EF-06 et non celui de "
          "l'API interactive.")
