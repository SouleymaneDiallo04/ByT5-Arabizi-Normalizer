"""Ce que vaudraient des paires fabriquees a partir de darija en ecriture arabe.

Des centaines de milliers de phrases de darija marocain existent en ecriture
arabe, sans arabizi en face : lid-dataset-base (AtlasIA) et fw-darija
(Sawalni). Fabriquer leur arabizi en ferait des paires d'entrainement. Avant
d'y consacrer du GPU, deux mesures, sans modele a entrainer :

  couverture  les mots que le modele rate faute de les avoir jamais vus
              figurent-ils dans ces textes ? Le budget d'erreur du 12 novembre
              attribue une part des mots faux a des formes absentes de
              l'entrainement ; des textes nouveaux ne peuvent corriger que
              celles-la, et seulement s'ils les contiennent.
  regles      un arabizi fabrique sans GPU, par lexique et regles lettre a
              lettre, ressemble-t-il a l'arabizi ecrit par des humains ?

Mots : normalisation d'evaluate.py, ponctuation retiree, hamzas strictes comme
la metrique. Un mot faux est un mot de la reference absent de la prediction
(difference de multiensembles).

Le jeu de test sert ici de diagnostic, comme pour le budget d'erreur : rien n'y
est choisi ni regle. Toute phrase candidate de meme cle qu'une phrase du test
est retiree avant comptage, et leur nombre est affiche.

"Jamais vu" se mesure contre le corpus du projet seul, puis contre le corpus et
tout NileChat. Le modele n'ayant vu qu'un tirage de NileChat, la verite est
entre les deux. NileChat est dans un depot prive de l'equipe ; sans acces,
--sans-nilechat ne donne que la premiere borne.

Usage, depuis la racine du depot :
    python -m evaluation.textes_darija byt5_melange3.csv
    python -m evaluation.textes_darija byt5_melange3.csv --sans-nilechat
"""

import argparse
import collections
import glob
import itertools
import random
import re

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

from donnees.normalisation import TABLE_PONCT, cle_arabe
from evaluation.evaluate import distance, normaliser

DEPOT_DONNEES = "sawti-nlp/darija-arabizi-corpus"
NILECHAT = ("sawti-nlp/nilechat-arabizi-pairs", "nilechat_translit.csv")
LID = ("atlasia/lid-dataset-base", "data/*.parquet")
FW = ("sawalni-ai/fw-darija", "ary_Arab/*.parquet")

ARABE = re.compile(r"[ء-يٱ-ۓ]")
FIN_PHRASE = re.compile(r"(?<=[.!?؟۔])\s+|\n+")
# Marqueurs de extract_nilechat.py : propres au darija, absents de l'arabe standard.
DARIJA = re.compile("ديال|اللي|كاين|هاد|باش|دابا|بزاف|واش|شنو|غادي|علاش|فين|"
                    "بغيت|مزيان|واخا|حيت|والو|ماشي|شوية|كيفاش")
MAX_CIBLE = 288
# Sources de lid-dataset-base deja presentes dans nos donnees, sous un autre nom.
DEJA_NOTRES = ("DODa-audio", "doda_audio", "nilechat")
REGIONS = ["Marrakech", "Casablanca", "Rabat", "Fes", "Generic", "Agadir",
           "Tetouan", "Nador", "Guelmim"]


def mots(texte):
    """Mots en ecriture arabe, normalises comme la metrique, sans ponctuation."""
    return [m for m in normaliser(texte).translate(TABLE_PONCT).split()
            if ARABE.search(m)]


def mots_faux(reference, prediction):
    """Mots de la reference absents de la prediction, repetitions comprises."""
    faux = collections.Counter(mots(reference)) - collections.Counter(mots(prediction))
    return list(faux.elements())


def mots_latins(texte):
    t = str(texte).lower().translate(TABLE_PONCT)
    return [re.sub(r"[^a-z0-9]", "", m) for m in t.split() if re.search(r"[a-z]", m)]


CONSONNES = {"ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "7", "خ": "kh", "د": "d",
             "ذ": "d", "ر": "r", "ز": "z", "س": "s", "ش": "ch", "ص": "s", "ض": "d",
             "ط": "t", "ظ": "d", "ع": "3", "غ": "gh", "ف": "f", "ق": "9", "ك": "k",
             "ل": "l", "م": "m", "ن": "n", "ه": "h", "ڭ": "g", "گ": "g", "ڤ": "v",
             "پ": "p", "ء": "", "ئ": "", "ؤ": "o"}


def regles(mot):
    """Arabizi d'un mot, lettre a lettre, aux conventions de messagerie marocaines.

    Les voyelles breves, que l'ecriture arabe n'ecrit pas, manquent ; et ك
    reste k meme la ou il se prononce g. Ce sont les deux limites que la mesure
    doit chiffrer.
    """
    sortie = []
    for i, c in enumerate(mot):
        if c in "اأآ":
            sortie.append("a")
        elif c == "إ":
            sortie.append("i")
        elif c in "ةى":
            sortie.append("a")
        elif c == "و":
            consonne = i == 0 or (i + 1 < len(mot) and mot[i + 1] in "اي")
            sortie.append("w" if consonne else "o")
        elif c == "ي":
            consonne = i == 0 or (i > 0 and mot[i - 1] in "او")
            sortie.append("y" if consonne else "i")
        else:
            sortie.append(CONSONNES.get(c, ""))
    return "".join(sortie)


def lexique(paires):
    """Graphies arabizi de chaque mot arabe, sur les paires d'egal nombre de mots."""
    graphies = collections.defaultdict(collections.Counter)
    for arabizi, arabe in paires:
        a, b = mots(arabe), mots_latins(arabizi)
        if a and len(a) == len(b):
            for x, y in zip(a, b):
                if y:
                    graphies[x][y] += 1
    return graphies


def phrases_lid(dossier):
    """Lignes de darija marocain en ecriture arabe, hors sources deja chez nous."""
    for f in sorted(glob.glob(f"{dossier}/data/*.parquet")):
        t = pq.read_table(f, columns=["src", "text", "dialect", "has_latin_script"])
        t = t.filter(pc.and_(pc.equal(t["dialect"], "moroccan_darija"),
                             pc.equal(t["has_latin_script"], "false")))
        for src, texte in zip(t["src"].to_pylist(), t["text"].to_pylist()):
            if not any(s in src for s in DEJA_NOTRES):
                yield "lid:" + src.split("|")[0], texte


def phrases_fw(dossier):
    """Pages web classees darija, decoupees en phrases."""
    for f in sorted(glob.glob(f"{dossier}/ary_Arab/*.parquet")):
        for doc in pq.read_table(f, columns=["text"])["text"].to_pylist():
            for phrase in FIN_PHRASE.split(doc or ""):
                if ARABE.search(phrase):
                    yield "fw-darija", phrase.strip()


def pourcent(a, b):
    return f"{100 * a / max(b, 1):.1f}%"


def couverture(predictions, train, avec_nilechat):
    """Volume des candidats, et part des mots jamais vus qu'ils contiennent."""
    ref_region, faux_region = collections.defaultdict(list), collections.defaultdict(list)
    for _, x in predictions.iterrows():
        ref_region[x["annotator_dialect"]] += mots(x["darija_norm"])
        faux_region[x["annotator_dialect"]] += mots_faux(x["darija_norm"], x["prediction"])
    W = {m for l in ref_region.values() for m in l}
    cles_test = {cle_arabe(x) for x in predictions["darija"]}
    print(f"test : {len(predictions)} phrases, {sum(map(len, ref_region.values()))} "
          f"mots de reference, {sum(map(len, faux_region.values()))} faux")

    # Ce que le modele a vu : seuls les mots du test sont retenus, et les cles
    # de phrase sous forme de hachage, pour tenir en memoire.
    vus_train, vus_nile, cles_vues = set(), set(), set()
    for t in train["darija_norm"].astype(str):
        vus_train.update(m for m in mots(t) if m in W)
        cles_vues.add(hash(cle_arabe(t)))
    if avec_nilechat:
        chemin = hf_hub_download(*NILECHAT, repo_type="dataset")
        for bloc in pd.read_csv(chemin, usecols=["darija"], chunksize=200_000):
            for t in bloc["darija"].astype(str):
                vus_nile.update(m for m in mots(t) if m in W)
                cles_vues.add(hash(cle_arabe(t)))

    lid = snapshot_download(LID[0], repo_type="dataset", allow_patterns=[LID[1]])
    fw = snapshot_download(FW[0], repo_type="dataset", allow_patterns=[FW[1]])
    compte = collections.defaultdict(collections.Counter)
    volume = collections.defaultdict(collections.Counter)
    exemples = collections.defaultdict(list)
    for source, texte in itertools.chain(phrases_lid(lid), phrases_fw(fw)):
        v = volume[source]
        v["lignes"] += 1
        cle = cle_arabe(texte)
        if cle in cles_test:
            v["retirees, test"] += 1
            continue
        propre = normaliser(texte)
        for m in mots(propre):
            if m in W:
                compte[source][m] += 1
        courte = len(propre.encode()) <= MAX_CIBLE
        darija = bool(DARIJA.search(propre))
        neuve = hash(cle) not in cles_vues
        v["courtes"] += courte
        v["darija"] += darija
        v["deja vues"] += not neuve
        v["utilisables"] += courte and darija and neuve
        if courte and len(exemples[source]) < 400:
            exemples[source].append((darija, propre))

    champs = ["lignes", "retirees, test", "courtes", "darija", "deja vues", "utilisables"]
    print("\nVolume. courtes : tiennent dans les 288 octets du modele ; darija : "
          "portent un marqueur ;\nutilisables : courtes, darija et absentes du "
          "train" + (" et de NileChat" if avec_nilechat else ""))
    sources = sorted(volume, key=lambda s: -volume[s]["lignes"])
    print(f"{'source':38}" + "".join(f"{c:>15}" for c in champs))
    total = collections.Counter()
    for s in sources:
        total.update(volume[s])
        print(f"{s:38}" + "".join(f"{volume[s][c]:>15}" for c in champs))
    print(f"{'total':38}" + "".join(f"{total[c]:>15}" for c in champs))

    candidats = collections.Counter()
    for c in compte.values():
        candidats.update(c)
    lid_seul = collections.Counter()
    for s, c in compte.items():
        if s.startswith("lid:"):
            lid_seul.update(c)

    vus = vus_train | vus_nile
    borne = "train et NileChat" if avec_nilechat else "train"
    print(f"\nMots faux du modele. jamais vus : absents du {borne} ; les colonnes "
          f"suivantes\nsont des parts de ces mots jamais vus")
    print(f"{'region':11}{'faux':>6}{'jamais vus':>12}{'(train seul)':>14}"
          f"{'dans lid':>10}{'lid ou fw':>11}{'3 fois ou +':>13}")
    for r in REGIONS + ["tout"]:
        faux = (faux_region[r] if r != "tout"
                else [m for l in faux_region.values() for m in l])
        jamais = [m for m in faux if m not in vus]
        print(f"{r:11}{len(faux):>6}"
              f"{len(jamais):>6} {pourcent(len(jamais), len(faux)):>6}"
              f"{pourcent(sum(m not in vus_train for m in faux), len(faux)):>14}"
              f"{pourcent(sum(lid_seul[m] > 0 for m in jamais), len(jamais)):>10}"
              f"{sum(candidats[m] > 0 for m in jamais):>5} "
              f"{pourcent(sum(candidats[m] > 0 for m in jamais), len(jamais)):>6}"
              f"{sum(candidats[m] >= 3 for m in jamais):>6} "
              f"{pourcent(sum(candidats[m] >= 3 for m in jamais), len(jamais)):>6}")

    print("\nFormes jamais vues des regions du sud et du nord, avec leurs "
          "occurrences dans les candidats")
    for r in ["Guelmim", "Tetouan", "Nador", "Agadir"]:
        jamais = collections.Counter(m for m in faux_region[r] if m not in vus)
        print(f"  {r} : " + ", ".join(f"{m} ({candidats[m]})"
                                      for m, _ in jamais.most_common(15)))

    print("\nEchantillons, au hasard, de phrases courtes de chaque source")
    hasard = random.Random(42)
    for s in sources:
        print(f"  --- {s}")
        for darija, p in hasard.sample(exemples[s], min(5, len(exemples[s]))):
            print(f"    [{'darija' if darija else '  ?   '}] {p[:120]}")


def realisme(train, test):
    """Arabizi par lexique ou par regles, contre l'arabizi humain du test, mot a mot."""
    graphies = lexique(zip(train["transliteration"], train["darija_norm"]))
    classes = {c: collections.Counter() for c in ("connus", "inconnus")}
    for arabizi, arabe in zip(test["transliteration"], test["darija_norm"]):
        a, b = mots(arabe), mots_latins(arabizi)
        if not a or len(a) != len(b):
            continue
        for x, humain in zip(a, b):
            if not humain:
                continue
            if x in graphies:
                c, produit = classes["connus"], graphies[x].most_common(1)[0][0]
            else:
                c, produit = classes["inconnus"], regles(x)
            c["erreurs"] += distance(produit, humain)
            c["caracteres"] += len(humain)
            c["exacts"] += produit == humain
            c["mots"] += 1
    print(f"\nArabizi fabrique, contre l'arabizi humain du test, mot a mot "
          f"({len(graphies)} mots arabes au lexique)")
    print("  connus : graphie la plus frequente du lexique, soit la variation "
          "entre scripteurs ;\n  inconnus : regles lettre a lettre")
    print(f"  {'mots':10}{'nombre':>8}{'CER':>8}{'exacts':>9}")
    for nom, c in classes.items():
        print(f"  {nom:10}{c['mots']:>8}{c['erreurs'] / max(c['caracteres'], 1):>8.3f}"
              f"{pourcent(c['exacts'], c['mots']):>9}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("predictions", help="sortie d'evaluation du modele, sur test_real.csv")
    p.add_argument("--sans-nilechat", action="store_true",
                   help="sans acces au depot prive : borne du seul corpus du projet")
    a = p.parse_args()
    predictions = pd.read_csv(a.predictions)
    split = lambda f: pd.read_csv(hf_hub_download(DEPOT_DONNEES, f"splits/{f}",
                                                  repo_type="dataset"))
    train, test = split("train.csv"), split("test_real.csv")
    couverture(predictions, train, not a.sans_nilechat)
    realisme(train, test)


if __name__ == "__main__":
    main()
