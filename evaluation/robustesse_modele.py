"""Robustesse du modele servi a un arabizi ecrit autrement.

L'exigence ENF-08 (evaluation/test_robustesse.py) verifie qu'aucune entree
bruitee ne fait planter la baseline. Elle ne dit pas ce que le modele perd en
qualite quand l'arabizi n'est pas ecrit comme dans TerjamaBench. Ce script le
mesure : il bruite l'arabizi des 850 phrases, un seul bruit a la fois, et
compare phrase par phrase la sortie bruitee a la sortie propre.

Chaque bruit imite une habitude des vrais messages : majuscule du telephone,
lettres allongees, fautes de frappe, voyelles omises, conventions d'ecriture
d'un autre scripteur (sh ou x pour ch, 5 pour kh...). Le tirage est seme par
phrase et par bruit : deux executions bruitent pareil (ENF-04). Seules les
phrases qu'un bruit modifie entrent dans sa mesure.

Le modele passe par le chemin du service : nettoyer, puis Normaliseur, en int8
comme servi (QUANTIFIE=0 pour le fp32). Une entree bruitee que l'API
refuserait, plus de MAX_OCTETS octets, est comptee puis ecartee.

Deux bruits ajoutent ce que la reference ne porte pas : emojis et ponctuation.
Le modele n'est pas juge sur leur recopie. Pour eux, reference, sortie propre
et sortie bruitee sont comparees sans emojis, ou sans ponctuation.

Les resultats sont donnes sur le reglage et sur l'evaluation. Choisir quel
bruit traiter se fait sur le reglage ; l'evaluation ne sert qu'a mesurer.

Usage, depuis la racine du depot, dans l'image du service (README) :
    python -m evaluation.robustesse_modele            decode, ecrit robustesse.csv
    python -m evaluation.robustesse_modele --relire   relit robustesse.csv
"""

import argparse
import random
import re
import time
import unicodedata

import pandas as pd

from evaluation.apparier import signe
from evaluation.evaluate import normaliser
from evaluation.relecture import par_phrase
from service.normaliseur import MAX_OCTETS, nettoyer

FICHIER = "robustesse.csv"
LOT = 64
VOYELLES = set("aeiouAEIOU")
EMOJIS = ["😂", "🙏", "😍", "👍", "❤️"]
PONCTUATIONS = ["!!!", "???", "...", "?!", "!!"]
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


# --- Les bruits. Chacun prend une phrase et un tirage, rend la phrase bruitee.

def capitale(p, tirage):
    """Le telephone met une majuscule a la premiere lettre."""
    return p[:1].upper() + p[1:]


def majuscules(p, tirage):
    return p.upper()


def allonger(p, tirage):
    """Une voyelle repetee quatre fois dans un mot : bzaaaaf."""
    mots = p.split()
    choix = [i for i, m in enumerate(mots) if any(c in VOYELLES for c in m)]
    if not choix:
        return p
    i = tirage.choice(choix)
    m = mots[i]
    j = max(k for k, c in enumerate(m) if c in VOYELLES)
    mots[i] = m[:j] + m[j] * 4 + m[j + 1:]
    return " ".join(mots)


def ponctuer(p, tirage):
    return p.rstrip() + tirage.choice(PONCTUATIONS)


def emojier(p, tirage):
    return p.rstrip() + tirage.choice(["", " "]) + tirage.choice(EMOJIS)


def fauter(p, tirage):
    """Deux caracteres voisins echanges dans un mot d'au moins quatre."""
    mots = p.split()
    choix = [(i, j) for i, m in enumerate(mots) if len(m) >= 4
             for j in range(len(m) - 1)
             if m[j].isalnum() and m[j + 1].isalnum() and m[j] != m[j + 1]]
    if not choix:
        return p
    i, j = tirage.choice(choix)
    m = mots[i]
    mots[i] = m[:j] + m[j + 1] + m[j] + m[j + 2:]
    return " ".join(mots)


def sans_voyelles(p, tirage):
    """Les voyelles internes d'un mot omises, comme en SMS : mzyan devient mzyn."""
    mots = p.split()
    choix = [i for i, m in enumerate(mots)
             if len(m) >= 4 and any(c in VOYELLES for c in m[1:-1])]
    if not choix:
        return p
    i = tirage.choice(choix)
    m = mots[i]
    mots[i] = m[0] + "".join(c for c in m[1:-1] if c not in VOYELLES) + m[-1]
    return " ".join(mots)


def coller(p, tirage):
    """Une espace oubliee entre deux mots."""
    mots = p.split()
    if len(mots) < 2:
        return p
    i = tirage.randrange(len(mots) - 1)
    return " ".join(mots[:i] + [mots[i] + mots[i + 1]] + mots[i + 2:])


def convention(motif, remplacement):
    """Un autre scripteur, une autre convention, appliquee partout."""
    def bruit(p, tirage):
        return re.sub(motif, remplacement, p, flags=re.IGNORECASE)
    return bruit


def chiffre(c, lettre):
    """Un chiffre-lettre remplace par une lettre, dans les mots seulement :
    le 7 de « 7 ayam » est un nombre et reste."""
    return convention(rf"(?<=[A-Za-z]){c}|{c}(?=[A-Za-z])", lettre)


def sans_emojis(t):
    return re.sub(r"\s+", " ", EMOJI.sub("", t)).strip()


def sans_ponctuation(t):
    t = "".join(c for c in t if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", t).strip()


# nom, bruit, transformation appliquee avant de comparer
BRUITS = [
    ("capitale", capitale, None),
    ("majuscules", majuscules, None),
    ("allongement", allonger, None),
    ("ponctuation", ponctuer, sans_ponctuation),
    ("emoji", emojier, sans_emojis),
    ("faute de frappe", fauter, None),
    ("voyelles omises", sans_voyelles, None),
    ("mots colles", coller, None),
    ("ch en sh", convention("ch", "sh"), None),
    ("ch en x", convention("ch", "x"), None),
    ("ou en u", convention("ou", "u"), None),
    ("kh en 5", convention("kh", "5"), None),
    ("9 en q", chiffre("9", "q"), None),
    ("7 en h", chiffre("7", "h"), None),
]
TRANSFORMATIONS = {nom: t for nom, _, t in BRUITS}


def bruiter(test):
    """Une ligne par phrase et par bruit qui la modifie, plus la phrase propre."""
    lignes = []
    for x in test.itertuples():
        texte = str(x.transliteration)
        lignes.append({"id": x.id, "bruit": "propre", "entree": texte, "refusee": False})
        for nom, bruit, _ in BRUITS:
            b = bruit(texte, random.Random(f"{nom}|{x.id}"))
            if nettoyer(b) == nettoyer(texte):
                continue
            lignes.append({"id": x.id, "bruit": nom, "entree": b,
                           "refusee": len(nettoyer(b).encode("utf-8")) > MAX_OCTETS})
    return pd.DataFrame(lignes)


def decoder(entrees):
    """Sorties du modele servi pour chaque entree distincte, par lots."""
    from service.normaliseur import Normaliseur

    n = Normaliseur()
    print(f"modele : {n.nom}@{n.revision[:7]} en {n.variante}")
    distinctes = list(dict.fromkeys(entrees))
    sorties, debut = {}, time.time()
    for i in range(0, len(distinctes), LOT):
        lot = distinctes[i:i + LOT]
        for e, (s, _, _) in zip(lot, n.normaliser([nettoyer(e) for e in lot])):
            sorties[e] = s
        fait = i + len(lot)
        print(f"  {fait}/{len(distinctes)}, {(time.time() - debut) / fait:.2f} s par phrase",
              flush=True)
    return [sorties[e] for e in entrees]


def mesurer(d, transformation=None):
    """CER propre et bruite, et bilan apparie, sur les phrases d'un bruit.

    d porte darija_norm, propre et bruitee, une ligne par phrase.
    """
    t = transformation or (lambda x: x)
    ref, a, b = (d[c].map(lambda x: t(normaliser(str(x))))
                 for c in ("darija_norm", "propre", "bruitee"))
    pa, pb = par_phrase(ref, a), par_phrase(ref, b)
    longueur = max(int(pa["longueur"].sum()), 1)
    mieux, pire = int((pb["erreurs"] < pa["erreurs"]).sum()), int((pb["erreurs"] > pa["erreurs"]).sum())
    perdues = int((pa["exacte"] & ~pb["exacte"]).sum())
    gagnees = int((~pa["exacte"] & pb["exacte"]).sum())
    return {"phrases": len(d),
            "cer_propre": pa["erreurs"].sum() / longueur,
            "cer_bruite": pb["erreurs"].sum() / longueur,
            "changees": int((a.values != b.values).sum()),
            "mieux": mieux, "pire": pire, "p": signe(mieux, pire),
            "perdues": perdues, "gagnees": gagnees, "p_exactes": signe(gagnees, perdues)}


def tableau(resultats, partie):
    d = resultats[resultats["partie"] == partie]
    propre = d[d["bruit"] == "propre"].set_index("id")
    tout = mesurer(propre.assign(propre=propre["sortie"], bruitee=propre["sortie"]))
    print(f"\n== {partie}, {len(propre)} phrases ; CER propre de toutes : "
          f"{tout['cer_propre']:.4f}")
    print(f"{'bruit':<17}{'phrases':>8}{'refusees':>9}{'CER propre':>11}{'bruite':>8}"
          f"{'ecart':>8}{'changees':>9}{'pires/meill.':>13}{'p':>7}{'exactes -/+':>12}")
    for nom, _, t in BRUITS:
        x = d[d["bruit"] == nom]
        refusees = int(x["refusee"].sum())
        x = x[~x["refusee"]].set_index("id")
        if x.empty:
            continue
        m = mesurer(x.assign(propre=propre.loc[x.index, "sortie"],
                             bruitee=x["sortie"]), t)
        ecart = (m["cer_bruite"] - m["cer_propre"]) / max(m["cer_propre"], 1e-9)
        print(f"{nom:<17}{m['phrases']:>8}{refusees:>9}{m['cer_propre']:>11.4f}"
              f"{m['cer_bruite']:>8.4f}{100 * ecart:>+7.0f}%{m['changees']:>9}"
              f"{m['pire']:>7}/{m['mieux']:<5}{m['p']:>7.3f}"
              f"{m['perdues']:>7}/{m['gagnees']:<4}")


def exemples(resultats, partie, n=2):
    """Les phrases que chaque bruit degrade le plus, pour lire ce qui se passe."""
    from evaluation.evaluate import distance

    d = resultats[(resultats["partie"] == partie) & ~resultats["refusee"]]
    propre = d[d["bruit"] == "propre"].set_index("id")
    print(f"\n== exemples, {partie}")
    for nom, _, t in BRUITS:
        t = t or (lambda x: x)
        x = d[d["bruit"] == nom].set_index("id")
        if x.empty:
            continue
        ref = x["darija_norm"].map(lambda r: t(normaliser(str(r))))
        ea = [distance(t(normaliser(str(s))), r) for s, r in zip(propre.loc[x.index, "sortie"], ref)]
        eb = [distance(t(normaliser(str(s))), r) for s, r in zip(x["sortie"], ref)]
        ecart = pd.Series([b - a for a, b in zip(ea, eb)], index=x.index).sort_values()
        print(f"-- {nom}")
        for i in ecart.index[::-1][:n]:
            if ecart[i] <= 0:
                break
            print(f"  {ecart[i]:+d}  {propre.loc[i, 'entree']}  ->  {x.loc[i, 'entree']}\n"
                  f"       propre : {propre.loc[i, 'sortie']}\n       bruite : {x.loc[i, 'sortie']}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--relire", action="store_true", help=f"relit {FICHIER} sans decoder")
    a = p.parse_args()

    if not a.relire:
        from huggingface_hub import hf_hub_download

        from donnees.decouper_test import DEPOT, TEST, prelever
        test = pd.read_csv(hf_hub_download(DEPOT, f"splits/{TEST}", repo_type="dataset"))
        reglage, _ = prelever(test)
        test["partie"] = ["reglage" if i in reglage.index else "evaluation" for i in test.index]
        lignes = bruiter(test)
        print(f"{len(lignes)} entrees, dont {int(lignes['refusee'].sum())} que l'API refuserait")
        a_decoder = lignes[~lignes["refusee"]]
        lignes.loc[a_decoder.index, "sortie"] = decoder(a_decoder["entree"].tolist())
        colonnes = ["id", "partie", "darija_norm", "topic", "annotator_dialect", "mot_isole"]
        lignes = lignes.merge(test[colonnes], on="id")
        lignes.to_csv(FICHIER, index=False, encoding="utf-8-sig")
        print(f"ecrit dans {FICHIER}")

    resultats = pd.read_csv(FICHIER, keep_default_na=False)
    resultats["refusee"] = resultats["refusee"].astype(str) == "True"
    for partie in ("reglage", "evaluation"):
        tableau(resultats, partie)
    exemples(resultats, "reglage")


if __name__ == "__main__":
    main()
