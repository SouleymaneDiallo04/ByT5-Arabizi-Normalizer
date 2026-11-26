"""Extraction de atlasia/moul_sout_transliteration_71k.

De la parole retranscrite en darija, avec son arabizi : 71 112 lignes, mais
35 546 phrases arabes seulement. Presque chaque phrase y figure deux fois, avec
deux arabizi ecrits differemment (lmo5alef et lmokhalif, b chwiya et bchwiya),
soit 68 347 paires distinctes. Rien ne dit s'ils viennent de deux annotateurs
ou d'un generateur.

Mesure sur le jeu brut (journal du 25 novembre) :

  - 24 % des lignes gardent du latin cote arabe, souvent des phrases entieres
    en francais ou en anglais. Elles ont d'abord ete conservees, faute de
    preuve qu'elles nuisent. La preuve est venue (journal du 26 novembre) :
    entraine avec elles, le modele recopie le francais, 25 phrases
    d'evaluation en latin au lieu de 1. Elles sont donc ecartees par defaut ;
    --avec-latin les garde, pour rejouer cette experience. Celles de
    daa-pairs restent : le melange 3, entraine avec, ne recopiait pas ;
  - 5 860 lignes depassent les 192 octets d'entree ou les 288 de cible.
    Conservees aussi : l'entrainement les coupe, et la coupe garde 86 % de la
    phrase en mediane, avec 1,1 mot de fin sans equivalent en face ;
  - 42 lignes recoupent le jeu de test : ajouter_source.py et melanger.py les
    retirent, comme pour toute source.

Le depot porte deux autres configurations. moul_sout_transliteration contient
exactement les memes paires ; synthetic_darija-open-dataset est doda_synthetic,
deja dans le corpus a 95 %. Aucune n'est extraite.

ATTENTION. La fiche du depot ne dit ni la provenance ni la licence. Comme pour
NileChat, le fichier produit ne doit pas etre publie, ni verse dans le corpus
du projet : seul ce script l'est. L'acces au depot se demande sur sa page et
AtlasIA le valide a la main.

Usage, depuis le dossier des donnees :
    python -m donnees.extraction.extract_moul_sout
"""

import argparse
import re
import unicodedata

import pandas as pd
from huggingface_hub import get_token, hf_hub_download

LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
SOURCE = "atlasia/moul_sout_transliteration_71k"
FICHIER = "data/train-00000-of-00001.parquet"
SORTIE = "moul_sout_translit.csv"


def a_du_latin(texte):
    """Vrai si le texte porte une lettre latine, accentuee ou non, apres NFKC."""
    return bool(LATIN.search(unicodedata.normalize("NFKC", str(texte))))


def extraire(brut, sans_latin=True):
    """Colonnes du corpus, lignes a cote vide retirees ; par defaut, lignes a
    latin cote arabe retirees aussi (journal du 26 novembre).

    L'identifiant du depot n'est pas unique : une phrase et ses deux arabizi le
    partagent. Le rang de la ligne dans la phrase le complete. Il est calcule
    avant le filtre du latin, pour qu'une paire garde le meme identifiant avec
    ou sans lui.
    """
    d = brut.dropna(subset=["latin_text", "arabic_text"])
    d = d[(d["latin_text"].astype(str).str.strip() != "")
          & (d["arabic_text"].astype(str).str.strip() != "")]
    d = d.assign(rang=d.groupby("sentence_id").cumcount())
    if sans_latin:
        d = d[~d["arabic_text"].map(a_du_latin)]
    return pd.DataFrame({
        "id": [f"moul-sout_{s:05d}_{r}" for s, r in zip(d["sentence_id"], d["rang"])],
        "darija": d["arabic_text"].values,
        "transliteration": d["latin_text"].values,
        "source": SOURCE,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--avec-latin", action="store_true",
                   help="garder les lignes a latin cote arabe, comme la premiere "
                        "experience du 26 novembre")
    a = p.parse_args()
    if not get_token():
        raise SystemExit("Aucun jeton Hugging Face. Demander l'acces sur la page "
                         f"de {SOURCE}, puis : hf auth login")
    brut = pd.read_parquet(hf_hub_download(SOURCE, FICHIER, repo_type="dataset"))
    complet = extraire(brut, sans_latin=False)
    out = complet if a.avec_latin else extraire(brut)
    out.to_csv(SORTIE, index=False, encoding="utf-8-sig")
    print(f"{len(brut)} lignes lues, {len(out)} conservees dans {SORTIE}")
    print(f"{len(brut) - len(complet)} ecartees pour un cote vide")
    if not a.avec_latin:
        print(f"{len(complet) - len(out)} ecartees pour du latin cote arabe")
    print(f"{out['darija'].nunique()} phrases arabes distinctes, "
          f"{len(out[['darija', 'transliteration']].drop_duplicates())} paires distinctes")
    print("Ce fichier ne doit pas etre publie : la licence de la source est inconnue.")
