import pandas as pd
import re
import unicodedata
from huggingface_hub import hf_hub_download

CORPUS = "darija_all.csv"
DATASET = "sawti-nlp/darija-arabizi-corpus"
TERJAMA = "test_candidates/terjamabench/data/test-00000-of-00001.parquet"
OUT_TEST = "test_real.csv"
OUT_TRAIN = "train.csv"
OUT_DEV = "dev.csv"
PART_DEV = 0.05
GRAINE = 42

# Signes de vocalisation et tatweel : TerjamaBench n'en porte pas (0,2 % des mots),
# alors que doda_synthetic en porte 21 %. Les laisser ferait compter chaque
# diacritique produit comme une erreur de caractère, sans erreur de langue.
VOCALISATION = [chr(c) for c in range(0x064B, 0x0656)] + ["ـ", "ٰ"]
TABLE_VOCAL = {ord(c): None for c in VOCALISATION}
# TerjamaBench écrit le point d'interrogation en arabe dans 100 % des cas,
# DODa-10K en latin dans 97 % des cas.
PONCTUATION_ARABE = {ord("?"): "؟", ord(";"): "؛"}

PONCT_CLE = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "،؛؟٪٫٬٭۔"
    "«»‘’“”…–—"
)
TABLE_PONCT = {ord(c): " " for c in PONCT_CLE}
EQUIVALENCES = [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")]


def normaliser_cible(texte):
    # Ce que le modèle devra produire. Seules les différences non débattues sont
    # corrigées ici ; la convention orthographique proprement dite reste à arbitrer.
    t = unicodedata.normalize("NFKC", str(texte)).translate(TABLE_VOCAL)
    return re.sub(r"\s+", " ", t.translate(PONCTUATION_ARABE)).strip()


def cle_arabe(texte):
    t = unicodedata.normalize("NFKC", str(texte)).translate(TABLE_PONCT).translate(TABLE_VOCAL)
    for avant, apres in EQUIVALENCES:
        t = t.replace(avant, apres)
    return re.sub(r"\s+", " ", t).strip()


def cle_latine(texte):
    t = unicodedata.normalize("NFKC", str(texte)).lower().translate(TABLE_PONCT)
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    try:
        corpus = pd.read_csv(CORPUS)
    except FileNotFoundError:
        raise SystemExit(f"{CORPUS} absent : lancer d'abord merge_all.py")

    # 1. Jeu de test : TerjamaBench, le seul arabizi écrit par des humains dont nous
    #    disposons. Les mots isolés sont marqués pour être évalués à part.
    t = pd.read_parquet(hf_hub_download(DATASET, TERJAMA, repo_type="dataset"))
    test = pd.DataFrame({
        "id": [f"terjama_{i:04d}" for i in range(len(t))],
        "darija": t["Darija"].map(lambda s: re.sub(r"\s+", " ", str(s)).strip()),
        "transliteration": t["Arabizi"].map(lambda s: re.sub(r"\s+", " ", str(s)).strip()),
        "source": "atlasia/TerjamaBench",
        "topic": t["topic"],
        "annotator_dialect": t["annotator_dialect"],
        "mot_isole": t["topic"].eq("single_words"),
    })
    test["darija_norm"] = test["darija"].map(normaliser_cible)
    test["key_ar"] = test["darija"].map(cle_arabe)
    test["key_la"] = test["transliteration"].map(cle_latine)

    # 2. Purge des fuites : toute ligne du corpus partageant un côté avec le test.
    avant = len(corpus)
    fuite = corpus["key_ar"].isin(set(test["key_ar"])) | corpus["key_la"].isin(set(test["key_la"]))
    corpus = corpus[~fuite]
    fuites = avant - len(corpus)

    # 3. Cible normalisée, le texte d'origine restant intact à côté.
    corpus["darija_norm"] = corpus["darija"].map(normaliser_cible)

    # 4. Découpage sur key_ar et non sur les lignes : une phrase présente dans deux
    #    sources doit rester d'un seul côté, sinon le modèle est évalué sur du déjà vu.
    #    Le dev écarte les lignes ambiguës, sa référence devant être unique.
    propre = corpus.groupby("key_ar")[["conflit", "suspect"]].max().sum(axis=1).eq(0)
    eligibles = propre[propre].index.to_series()
    n_dev = int(corpus["key_ar"].nunique() * PART_DEV)
    cles_dev = set(eligibles.sample(min(n_dev, len(eligibles)), random_state=GRAINE))

    dev = corpus[corpus["key_ar"].isin(cles_dev)]
    train = corpus[~corpus["key_ar"].isin(cles_dev)]

    for nom, d in [(OUT_TEST, test), (OUT_DEV, dev), (OUT_TRAIN, train)]:
        d.to_csv(nom, index=False, encoding="utf-8-sig")

    print(f"{fuites} lignes retirées du corpus pour recoupement avec le test")
    print(f"{OUT_TRAIN:14} {len(train):>7} lignes  ({train['key_ar'].nunique()} phrases distinctes)")
    print(f"{OUT_DEV:14} {len(dev):>7} lignes  ({dev['key_ar'].nunique()} phrases distinctes)")
    print(f"{OUT_TEST:14} {len(test):>7} lignes  (dont {int(test['mot_isole'].sum())} mots isolés)")
    print()
    print(f"phrases communes train / dev  : {len(set(train['key_ar']) & set(dev['key_ar']))}")
    print(f"phrases communes train / test : {len(set(train['key_ar']) & set(test['key_ar']))}")
    print(f"lignes ambiguës dans le dev   : {int((dev['conflit'] | dev['suspect']).sum())}")
    print(f"cibles modifiées par la normalisation : "
          f"{int((corpus['darija'] != corpus['darija_norm']).sum())} sur {len(corpus)}")
