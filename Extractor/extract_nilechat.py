"""Jointure du corpus NileChat : arabizi genere et source arabe, en paires.

\x60UBC-NLP/nilechat-arabizi-mor\x60 contient 1 450 337 documents en arabizi, mais sans
leur version arabe : ses colonnes sont \x60id\x60 et \x60text\x60, rien d'autre. Les sources
arabes dont il a ete tire existent separement, et les identifiants concordent au
caractere pres. La jointure a ete verifiee : quatre shards arabes sur quarante-six
couvrent deja 9,9 % d'un shard arabizi.

Ce sont des documents et non des phrases, de 1 868 et 3 401 octets en mediane.
Aucun ne tient dans les 192 octets du modele. Il faut donc les decouper, et c'est
le decoupage qui decide de tout : mesure sur 5 251 documents apparies, 86,5 %
ont le meme nombre de lignes des deux cotes, l'ecart median etant nul. Le modele
de translittération a preserve la structure, et l'alignement ligne a ligne suffit.

Rendement mesure : 5,4 paires exploitables par document, soit de l'ordre de 7,8
millions de paires sur l'ensemble.

ATTENTION. L'arabizi de ce corpus est **produit par un modele de langue**, pas ecrit par
des humains. C'est exactement la degradation automatique que le projet avait
ecartee par raisonnement pour Atlaset ; elle est ici mesurable, et c'est la
seule raison de s'y interesser. Ne jamais presenter ces paires comme de
l'arabizi reel.

ATTENTION. La source arabe est en **CC-BY-ND**, qui interdit de distribuer une oeuvre
derivee, et un corpus de paires en est une. L'arabizi est en CC-BY-NC, donc non
commercial. **Le fichier produit ne doit pas etre publie.** Seul ce script l'est,
comme pour toutes les autres sources du projet : quiconque a accepte les
conditions des depots le reconstruit en une commande.

Usage :
    python Extractor/extract_nilechat.py --apercu   un couple de shards, 5 min
    python Extractor/extract_nilechat.py            tout, 40 a 60 min
"""

import collections
import re
import sys

import pandas as pd
import requests
from huggingface_hub import get_token

ARABIZI = "UBC-NLP/nilechat-arabizi-mor"
# nilechat-fw-edu-mor, 5 247 498 documents, est ecarte : c'est du texte web
# educatif, massivement en arabe standard. L'apercu en a tire des citations
# scientifiques, des metadonnees de revue et des phrases ou le modele a traduit
# au lieu de translitterer, laissant une cible en arabe standard face a un
# arabizi darija. Tout le rebut venait de la. nilechat-lhv-mor est
# conversationnel, et toutes les bonnes paires en venaient.
ARABES = ["UBC-NLP/nilechat-lhv-mor"]
SORTIE = "nilechat_translit.csv"
COLONNES = ["id", "darija", "transliteration", "source"]

MAX_SOURCE, MAX_CIBLE = 192, 288
# Bande de rapport de longueur entre les deux cotes. L'arabe s'ecrit en deux
# octets par caractere mais s'ecrit plus court : la bande est donc large et
# asymetrique, et elle ne sert qu'a ecarter les lignes franchement desappariees.
RATIO_MIN, RATIO_MAX = 0.5, 1.6
APERCU_EXEMPLES = 30

LATIN = re.compile(r"[A-Za-z]")
ARABE = re.compile(r"[؀-ۿ]")
# Marqueurs propres au darija, absents de l'arabe standard : ce dernier ecrit
# الذي et non اللي, هذا et non هاد, لماذا et non علاش. Le filtre est
# volontairement severe. Il rejette aussi de bonnes phrases courtes qui n'en
# portent aucun, comme آمين الله يحفظ بلادنا, et c'est assume : on a des
# millions de candidats et un deficit de qualite, donc la precision prime sur
# le rappel. Les mots ambigus, خاص et عند qui existent en standard, sont exclus.
DARIJA = re.compile("ديال|اللي|كاين|هاد|باش|دابا|بزاف|واش|شنو|غادي|علاش|فين|"
                    "بغيت|مزيان|واخا|حيت|والو|ماشي|شوية|كيفاش")

MOTIFS = collections.Counter()


def entetes():
    jeton = get_token()
    if not jeton:
        raise SystemExit(
            "Aucun jeton Hugging Face. Ces depots demandent d'avoir accepte "
            "leurs conditions sur leur page, puis : hf auth login")
    return {"Authorization": f"Bearer {jeton}"}


def shards(jeu, h):
    r = requests.get("https://datasets-server.huggingface.co/parquet",
                     params={"dataset": jeu}, headers=h, timeout=120)
    r.raise_for_status()
    return [f["url"] for f in r.json()["parquet_files"] if f["split"] == "train"]


def lignes(texte):
    return [x.strip() for x in str(texte).split("\n") if x.strip()]


def apparier(latin, arabe):
    """Rend les lignes alignees d'un document, ou rien si la structure differe.

    Un document dont les deux cotes n'ont pas le meme nombre de lignes est
    ecarte sans discussion. On en perd 13,5 %, c'est mesure, et c'est le prix
    d'un alignement dont on peut repondre : deviner un decalage de lignes
    produirait des paires fausses qu'aucun controle ulterieur ne rattraperait.
    """
    g, d = lignes(latin), lignes(arabe)
    if len(g) != len(d) or not g:
        MOTIFS["document ecarte, structure differente"] += 1
        return []
    retenues = []
    for a, b in zip(g, d):
        oa, ob = len(a.encode()), len(b.encode())
        if oa > MAX_SOURCE or ob > MAX_CIBLE:
            MOTIFS["ligne trop longue"] += 1
        elif not (RATIO_MIN <= oa / max(ob, 1) <= RATIO_MAX):
            MOTIFS["longueurs desappariees"] += 1
        elif not LATIN.search(a) or not ARABE.search(b):
            # URL, nombres seuls, debris de mise en page : ce n'est pas une
            # paire de translitteration.
            MOTIFS["pas une paire d_ecritures"] += 1
        elif not DARIJA.search(b):
            MOTIFS["cible sans marqueur darija"] += 1
        else:
            MOTIFS["retenue"] += 1
            retenues.append((a, b))
    return retenues


if __name__ == "__main__":
    apercu = "--apercu" in sys.argv
    h = entetes()

    urls_lat = shards(ARABIZI, h)
    if apercu:
        urls_lat = urls_lat[:1]
    print(f"{len(urls_lat)} shards arabizi a charger en memoire")

    arabizi = {}
    for i, u in enumerate(urls_lat, 1):
        d = pd.read_parquet(u, columns=["id", "text"], storage_options=h)
        arabizi.update(zip(d["id"], d["text"]))
        print(f"  shard {i}/{len(urls_lat)} : {len(arabizi)} documents cumules")
    del d

    # En mode complet les paires sont ecrites au fil des shards et non
    # accumulees : 7,8 millions de tuples puis leur conversion en DataFrame
    # feraient un pic d'environ neuf gigaoctets, pour rien.
    paires, documents, structure, ecrites, premier = [], 0, 0, 0, True
    for jeu in ARABES:
        urls = shards(jeu, h)
        if apercu:
            urls = urls[:1]
        print(f"\n{jeu} : {len(urls)} shards")
        for i, u in enumerate(urls, 1):
            d = pd.read_parquet(u, columns=["id", "text"], storage_options=h)
            d = d[d["id"].isin(arabizi.keys())]
            documents += len(d)
            lot = []
            for ident, arabe in zip(d["id"], d["text"]):
                retenues = apparier(arabizi[ident], arabe)
                if retenues:
                    structure += 1
                for rang, (a, b) in enumerate(retenues):
                    lot.append((f"nilechat_{ident}_{rang:02d}", b, a, jeu))
            if apercu:
                paires.extend(lot)
            elif lot:
                pd.DataFrame(lot, columns=COLONNES).to_csv(
                    SORTIE, mode="w" if premier else "a", header=premier,
                    index=False, encoding="utf-8-sig")
                premier = False
            ecrites += len(lot)
            print(f"  shard {i}/{len(urls)} : {len(d)} apparies, "
                  f"{ecrites} paires cumulees")

    print(f"\n{documents} documents apparies par identifiant")
    print(f"{structure} ont la meme structure de lignes "
          f"({100 * structure / max(documents, 1):.1f} %)")
    print(f"{ecrites} paires retenues, soit "
          f"{ecrites / max(documents, 1):.1f} par document")
    lignes_vues = sum(n for m, n in MOTIFS.items() if "document" not in m)
    print()
    print(f"sort des {lignes_vues} lignes examinees")
    for motif, n in MOTIFS.most_common():
        if "document" in motif:
            continue
        print(f"  {motif:<32} {n:>8} {100 * n / max(lignes_vues, 1):>6.1f} %")

    if apercu:
        print(f"\n--- {APERCU_EXEMPLES} paires au hasard, a lire avant d'aller "
              f"plus loin ---")
        import random
        for identifiant, ar, lat, _ in random.Random(42).sample(
                paires, min(APERCU_EXEMPLES, len(paires))):
            print(f"\n  lat  {lat}")
            print(f"  ar   {ar}")
        print(f"\nRien n'est ecrit en mode apercu. Si ces paires tiennent, "
              f"relancer sans --apercu.")
        raise SystemExit

    out = pd.DataFrame(paires, columns=["id", "darija", "transliteration", "source"])
    out.to_csv(SORTIE, index=False, encoding="utf-8-sig")
    print(f"\n{SORTIE} ecrit : {len(out)} paires")
    print("Ce fichier ne doit pas etre publie : la source arabe est en CC-BY-ND.")
