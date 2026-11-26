"""Sensibilite du modele a la quantification, groupe de couches par groupe.

Le cadrage du projet demande un modele quantifie. Quantifier toutes les couches
lineaires en entier 8 bits multipliait pourtant le CER par 3,8 (cahier des
charges, ENF-02). Plutot que d'ecarter la quantification d'un bloc, on cherche
ou se loge la degradation. La famille T5 porte des activations extremes en
sortie de ses couches feed-forward (wo), au point que transformers garde ces
couches en pleine precision quand le reste du modele passe en fp16.

Deux etapes, qui ne se melangent pas :

  selection  chaque groupe de couches est quantifie seul, puis les groupes
             inoffensifs ensemble, sur les 172 phrases du jeu de reglage :
             choisir une configuration est un reglage, il ne se fait pas sur
             les phrases d'evaluation ;
  mesure     la configuration retenue est mesuree une seule fois sur les 678
             phrases d'evaluation, avec --final.

Processeur uniquement : c'est la que le service tourne, et la quantification
dynamique de torch n'existe que la.

Usage, depuis la racine du depot :
    python -m evaluation.quantification
    python -m evaluation.quantification --final enc_attention,dec_attention
"""

import argparse
import time

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from donnees.decouper_test import prelever
from evaluation.evaluate import mesurer, normaliser
from evaluation.mesures import DEPOT_DONNEES, latence, taille_mo
from modele.inference import predire
from modele.quantification import GROUPES, couches, quantifier
from service.config import MODELE, REVISION

# Un groupe dont la quantification seule degrade le CER de moins de 2 % est
# juge inoffensif, et rejoint la combinaison finale. Les configurations sont
# comparees sur les memes phrases : l'ecart ne doit rien au tirage.
TOLERANCE = 0.02
PHRASES_LATENCE = 50


def charger_modele():
    """Le modele servi, a la revision figee du service, en pleine precision."""
    return AutoModelForSeq2SeqLM.from_pretrained(MODELE, revision=REVISION).eval()


def evaluer(nom, groupes, par_groupe, tok, donnees):
    """CER, exactitude, taille et latence d'une configuration."""
    modele = charger_modele()
    noms = [n for g in groupes for n in par_groupe[g]]
    if noms:
        modele = quantifier(modele, noms)
    debut = time.perf_counter()
    pred, _ = predire(modele, tok, donnees["transliteration"].astype(str).tolist(), "cpu")
    duree = time.perf_counter() - debut
    m = mesurer(donnees["darija_norm"].map(normaliser), [normaliser(p) for p in pred])
    phrases = donnees["transliteration"].astype(str).tolist()[:PHRASES_LATENCE]
    lat = latence(modele, tok, phrases, "cpu", 1)
    return {"configuration": nom, "couches": len(noms), "taille_mo": taille_mo(modele),
            "cer": m["cer"], "exactitude": m["exactitude"],
            "latence_mediane": lat["mediane"], "latence_p90": lat["p90"],
            "duree_s": duree}


def afficher(ligne, reference):
    print(f"{ligne['configuration'][:44]:44} {ligne['couches']:>4} "
          f"{ligne['taille_mo']:>6.0f} {ligne['cer']:>8.4f} "
          f"{ligne['cer'] / reference['cer']:>6.2f}x "
          f"{100 * ligne['exactitude']:>6.1f}% "
          f"{ligne['latence_mediane']:>7.3f}s {ligne['latence_p90']:>6.3f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--final", help="groupes separes par des virgules, mesures une "
                                   "fois sur les 678 phrases d'evaluation")
    a = p.parse_args()

    print(f"modele : {MODELE}@{REVISION[:7]} | processeur, "
          f"{torch.get_num_threads()} fils | torch {torch.__version__}")
    tok = AutoTokenizer.from_pretrained(MODELE, revision=REVISION)
    par_groupe, hors_groupe = couches(charger_modele())
    for g, noms in par_groupe.items():
        print(f"  {g:15} {len(noms):3} couches")
    if hors_groupe:
        print(f"  hors groupe, jamais quantifiees : {hors_groupe}")

    test = pd.read_csv(hf_hub_download(DEPOT_DONNEES, "splits/test_real.csv",
                                       repo_type="dataset"))
    reglage, evaluation = prelever(test)

    if a.final:
        groupes = a.final.split(",")
        inconnus = sorted(set(groupes) - set(GROUPES))
        if inconnus:
            raise SystemExit(f"groupes inconnus : {inconnus} ; connus : {list(GROUPES)}")
        donnees, plan = evaluation, [("fp32", []), ("+".join(groupes), groupes)]
        print(f"\nmesure finale, {len(evaluation)} phrases d'evaluation")
    else:
        donnees = reglage
        plan = [("fp32", [])] + [(g, [g]) for g in GROUPES]
        plan += [("tout", list(GROUPES)),
                 ("tout sauf ffn_sortie", [g for g in GROUPES if not g.endswith("ffn_sortie")])]
        print(f"\nselection, {len(reglage)} phrases du jeu de reglage")

    print(f"\n{'configuration':44} {'couch':>4} {'Mo':>6} {'CER':>8} {'/fp32':>7} "
          f"{'exact.':>7} {'lat.med':>8} {'p90':>7}")
    lignes = []
    for nom, groupes in plan:
        lignes.append(evaluer(nom, groupes, par_groupe, tok, donnees))
        afficher(lignes[-1], lignes[0])

    if not a.final:
        seuil = lignes[0]["cer"] * (1 + TOLERANCE)
        inoffensifs = [l["configuration"] for l in lignes[1:1 + len(GROUPES)]
                       if l["cer"] <= seuil]
        print(f"\ngroupes inoffensifs (CER a moins de {100 * TOLERANCE:.0f} % du fp32) : "
              f"{inoffensifs or 'aucun'}")
        if len(inoffensifs) > 1:
            lignes.append(evaluer("ensemble : " + "+".join(inoffensifs), inoffensifs,
                                  par_groupe, tok, donnees))
            afficher(lignes[-1], lignes[0])

    sortie = "quantification_final.csv" if a.final else "quantification_reglage.csv"
    pd.DataFrame(lignes).to_csv(sortie, index=False)
    print(f"\nresultats dans {sortie}")


if __name__ == "__main__":
    main()
