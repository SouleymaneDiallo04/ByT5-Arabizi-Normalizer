"""Compare notre modele aux modeles publics de translitteration, sur nos phrases.

Chaque modele annonce ses chiffres sur son propre jeu, souvent en mots isoles :
ils ne se comparent ni entre eux ni aux notres. Ici, tous passent sur les memes
phrases de TerjamaBench et par la meme mesure, evaluation.evaluate.mesurer.

Trois precautions, sans lesquelles un modele paraitrait mauvais a tort :

  - Poids mal charges. Un T5 sauvegarde par transformers 4 avec
    tie_word_embeddings=False n'ecrit ses plongements qu'une fois, et
    transformers 5 reinitialise au hasard ceux qu'il ne trouve pas. Ils sont ici
    relies a la matrice partagee, et un modele a qui il manque d'autres poids
    est signale au lieu d'etre mesure.
  - Prefixe d'entree. Deux modeles n'en documentent aucun. Les prefixes
    plausibles sont essayes sur 40 phrases de reglage, et le meilleur est garde.
  - Reglages de generation. Chaque modele garde ceux de ses auteurs, faisceaux
    compris. Seule la longueur maximale est relevee, la valeur par defaut de
    transformers tronquant les phrases a 20 jetons.

Seuls des poids safetensors sont charges, et aucun code distant n'est execute.

Les scores de ces modeles ne disent rien de leur valeur sur leur propre domaine :
deux d'entre eux visent l'algerien ou l'arabe du Golfe, et non le darija marocain.

Usage, depuis la racine du depot, un GPU de preference :
    python -m evaluation.comparer_modeles           172 phrases de reglage
    python -m evaluation.comparer_modeles --final   678 phrases d'evaluation
    python -m evaluation.comparer_modeles --seulement atlasia   un seul modele
La commande --final relit les predictions ecrites par les autres.
"""

import argparse
import os
import re

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, model_info
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from donnees.decouper_test import charger, prelever
from evaluation.evaluate import mesurer, normaliser
from modele.inference import MAX_CIBLE, MAX_SOURCE
from service.config import MODELE, REVISION

TACHE_MT5 = "transliterate_latin_darija"
CANDIDATS = [
    {"nom": "melange 3 (nous)", "depot": MODELE, "revision": REVISION,
     "octets": True, "prefixes": [""]},
    # Entraine sur des mots : sur 172 phrases de reglage, son CER passe de 0,78 en
    # phrases entieres a 0,30 mot par mot et en minuscules. Son tokenizer ne
    # distingue pas une coupure dans un mot d'une limite de mots, d'ou les espaces
    # internes retires a la sortie.
    {"nom": "AtlasIA, Transformer (par mot)", "depot": "atlasia/Transliteration-Moroccan-Darija",
     "prefixes": [""], "par_mot": True, "minuscules": True, "sans_espaces": True},
    {"nom": "akachkach, ByT5", "depot": "akachkach/byt5-darija-transliteration-v1",
     "octets": True,
     "prefixes": ["", "transliterate: ", "transliterate Darija: ", "arabizi: ",
                  "to arabic: ", "translate Arabizi to Arabic: "]},
    # Sa configuration nomme MT5Tokenizer, que transformers 5 ne sait plus
    # instancier. Son spiece.model est identique octet pour octet a celui de
    # google/mt5-small, dont le tokenizer se charge.
    {"nom": "Haitam03, mT5", "depot": "Haitam03/darija-mt5-translation-transliteration",
     "depot_tokenizer": "google/mt5-small",
     "prefixes": ["", f"{TACHE_MT5}: ", f"{TACHE_MT5} ", f"<{TACHE_MT5}> ", f"[{TACHE_MT5}] ",
                  "transliterate latin darija: "]},
    {"nom": "Khenafif, mBART (algerien)",
     "depot": "Chaima-KHENAFIF/darija-latin-to-arabic-translator", "prefixes": [""]},
    {"nom": "masterofaudio, ByT5 (Golfe, par mot)",
     "depot": "masterofaudio2077/byt5-small-arabizi-word", "octets": True,
     "prefixes": ["to arabic: "], "par_mot": True},
]
LOT = 16
ESSAI_PREFIXE = 40
LONGUEUR_JETONS = 128
# Tete de BERT : ces deux poids sont lies a d'autres, et le fichier ne les ecrit pas.
LIENS = {"cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
         "cls.predictions.decoder.bias": "cls.predictions.bias"}


def reparer(modele, manquantes):
    """Relie les matrices partagees que le fichier n'a ecrites qu'une fois.

    Selon la version qui a sauvegarde le modele, la matrice est rangee sous le
    nom `shared` ou sous celui d'un des plongements : la premiere presente sert
    de source aux absentes. Renvoie les cles manquantes qui ne s'expliquent pas
    ainsi : le modele qui en a n'est pas celui que ses auteurs ont entraine.
    """
    hors_liens = []
    for cle in manquantes:
        suffixe = next((s for s in LIENS if cle.endswith(s)), None)
        if suffixe is None:
            hors_liens.append(cle)
            continue
        module, attribut = cle.rsplit(".", 1)
        source = cle[:-len(suffixe)] + LIENS[suffixe]
        setattr(modele.get_submodule(module), attribut, modele.get_parameter(source))
    manquantes = hors_liens

    coeur = modele.model if hasattr(getattr(modele, "model", None), "shared") else modele
    candidates = {
        "shared": getattr(coeur, "shared", None),
        "encoder.embed_tokens": getattr(getattr(coeur, "encoder", None), "embed_tokens", None),
        "decoder.embed_tokens": getattr(getattr(coeur, "decoder", None), "embed_tokens", None),
        "lm_head": getattr(modele, "lm_head", None),
    }
    matrices = {n: m for n, m in candidates.items() if m is not None}
    est_matrice = lambda cle: any(cle.endswith(n + ".weight") for n in matrices)
    absentes = [n for n in matrices if any(c.endswith(n + ".weight") for c in manquantes)]
    presentes = [n for n in matrices if n not in absentes]

    restantes = [c for c in manquantes if "final_logits_bias" not in c and not est_matrice(c)]
    if absentes and not presentes:
        return restantes + [c for c in manquantes if est_matrice(c)]
    for nom in absentes:
        matrices[nom].weight = matrices[presentes[0]].weight
    return restantes


def charger_modele(candidat, appareil):
    """Tokenizer et modele, en safetensors et sans code distant.

    Les cles sont lues dans le fichier lui-meme, et non dans le rapport de
    chargement de transformers, dont la forme change d'une version a l'autre.
    """
    from safetensors import safe_open

    depot, revision = candidat["depot"], candidat.get("revision")
    tok = AutoTokenizer.from_pretrained(candidat.get("depot_tokenizer", depot), revision=revision)
    modele = AutoModelForSeq2SeqLM.from_pretrained(depot, revision=revision,
                                                   use_safetensors=True)
    with safe_open(hf_hub_download(depot, "model.safetensors", revision=revision),
                   framework="pt") as f:
        cles = set(f.keys())
    # Une cle absente du fichier mais qui partage sa memoire avec une cle presente
    # est deja liee par transformers : seules les autres sont a reparer.
    etat = modele.state_dict()
    adresses = {etat[c].data_ptr() for c in cles if c in etat}
    restantes = reparer(modele, sorted(c for c in etat
                                       if c not in cles and etat[c].data_ptr() not in adresses))
    if restantes:
        raise RuntimeError(f"poids absents du fichier : {restantes[:4]}")
    return tok, modele.to(appareil).eval()


def reglages(candidat, modele, tok):
    """Longueurs d'entree et de generation adaptees au modele."""
    if candidat.get("par_mot") and candidat.get("octets"):
        return MAX_SOURCE, {}          # les auteurs fixent 40 octets de sortie
    if candidat.get("octets"):
        return MAX_SOURCE, {"max_length": MAX_CIBLE}
    # Un EncoderDecoderModel range cette limite dans ses deux sous-configurations.
    config = modele.config
    limites = [getattr(c, "max_position_embeddings", None)
               for c in (config, getattr(config, "encoder", None), getattr(config, "decoder", None))]
    limite = min([v for v in limites if v] or [LONGUEUR_JETONS])
    generation = {"max_length": min(LONGUEUR_JETONS, limite)}
    # Sans fin de sequence declaree, le modele ecrirait jusqu'a la longueur maximale.
    if modele.generation_config.eos_token_id is None and tok.eos_token_id is not None:
        generation["eos_token_id"] = tok.eos_token_id
    return min(MAX_SOURCE, limite), generation


def generer(modele, tok, textes, appareil, source, generation):
    sorties = []
    for depart in range(0, len(textes), LOT):
        entrees = tok(textes[depart:depart + LOT], return_tensors="pt", padding=True,
                      truncation=True, max_length=source)
        entrees.pop("token_type_ids", None)
        with torch.no_grad():
            s = modele.generate(**entrees.to(appareil), **generation)
        sorties.extend(tok.batch_decode(s, skip_special_tokens=True))
    return sorties


def predire(candidat, modele, tok, phrases, appareil, prefixe):
    """Une sortie par phrase. Un modele de mots traite chaque mot, puis recolle."""
    source, generation = reglages(candidat, modele, tok)
    if not candidat.get("par_mot"):
        return generer(modele, tok, [prefixe + p for p in phrases], appareil, source, generation)
    forme = str.lower if candidat.get("minuscules") else str
    mots = sorted({forme(m) for p in phrases for m in p.split()})
    sorties = generer(modele, tok, [prefixe + m for m in mots], appareil, source, generation)
    if candidat.get("sans_espaces"):
        sorties = [s.replace(" ", "") for s in sorties]
    table = dict(zip(mots, sorties))
    return [" ".join(table[forme(m)].strip() for m in p.split()) for p in phrases]


def choisir_prefixe(candidat, modele, tok, reglage, appareil):
    prefixes = candidat["prefixes"]
    if len(prefixes) == 1:
        return prefixes[0]
    essai = reglage.head(ESSAI_PREFIXE)
    phrases = essai["transliteration"].astype(str).tolist()
    references = essai["darija_norm"].map(normaliser).tolist()
    scores = {}
    for prefixe in prefixes:
        sortie = predire(candidat, modele, tok, phrases, appareil, prefixe)
        scores[prefixe] = mesurer(references, [normaliser(s) for s in sortie])["wer"]
    return min(scores, key=scores.get)


def fichier(candidat):
    return "comparaison_" + re.sub(r"[^a-z0-9]+", "_", candidat["nom"].lower()).strip("_") + ".csv"


def ligne(nom, d, extra=""):
    m = mesurer(d["darija_norm"].map(normaliser),
                d["prediction"].fillna("").astype(str).map(normaliser))
    print(f"{nom[:38]:38} {m['cer']:>7.4f} {m['wer']:>7.4f} {100 * m['exactitude']:>6.1f}%{extra}",
          flush=True)
    return m


def entete():
    print(f"\n{'modele':38} {'CER':>7} {'WER':>7} {'exact.':>7}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--final", action="store_true",
                   help="mesure les predictions deja ecrites sur les 678 phrases d'evaluation")
    p.add_argument("--seulement", help="ne relance que les modeles dont le nom contient ce texte")
    a = p.parse_args()

    test = charger("test_real.csv")
    reglage, evaluation = prelever(test)

    if a.final:
        print(f"evaluation, {len(evaluation)} phrases")
        entete()
        for c in CANDIDATS:
            if os.path.exists(fichier(c)):
                d = pd.read_csv(fichier(c))
                ligne(c["nom"], d.loc[evaluation.index])
        return

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"reglage, {len(reglage)} phrases, sur {appareil}")
    choisis = [c for c in CANDIDATS
               if not a.seulement or a.seulement.lower() in c["nom"].lower()]
    if not choisis:
        raise SystemExit(f"aucun modele ne correspond a {a.seulement!r}")
    for c in choisis:
        if os.path.exists(fichier(c)):
            os.remove(fichier(c))
    phrases = test["transliteration"].astype(str).tolist()
    entete()
    echecs = []
    for c in choisis:
        try:
            tok, modele = charger_modele(c, appareil)
            prefixe = choisir_prefixe(c, modele, tok, reglage, appareil)
            sortie = predire(c, modele, tok, phrases, appareil, prefixe)
            taille = sum(x.numel() * x.element_size() for x in modele.parameters()) / 1e6
            exemple = predire(c, modele, tok, ["chrit 3 tomobilat"], appareil, prefixe)[0]
            sha = c.get("revision") or model_info(c["depot"]).sha
        except Exception as erreur:
            echecs.append((c["nom"], type(erreur).__name__, str(erreur).split("\n")[0][:150]))
            print(f"{c['nom'][:38]:38} ECHEC, voir la fin", flush=True)
            continue
        d = test.copy()
        d["prediction"] = sortie
        d.to_csv(fichier(c), index=False, encoding="utf-8-sig")
        ligne(c["nom"], d.loc[reglage.index],
              f"  {taille:5.0f} Mo  @{sha[:7]}  prefixe {prefixe!r}  exemple : {exemple}")
        del modele
        if appareil == "cuda":
            torch.cuda.empty_cache()

    for nom, type_erreur, message in echecs:
        print(f"\nECHEC {nom} : {type_erreur} : {message}")


if __name__ == "__main__":
    main()
