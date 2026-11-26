import pandas as pd
import re
from huggingface_hub import hf_hub_download

CORPUS = "darija_all.csv"
DATASET = "sawti-nlp/darija-arabizi-corpus"
TERJAMA = "test_candidates/terjamabench/data/test-00000-of-00001.parquet"
OUT_TEST = "test_real.csv"
OUT_TRAIN = "train.csv"
OUT_DEV = "dev.csv"
PART_DEV = 0.05
GRAINE = 42

# Definies dans normalisation.py, sans dependance, pour le registre et
# l'interface de validation. Reexportees ici : melanger.py, ajouter_source.py
# et les tests les importent depuis ce module.
from donnees.normalisation import (  # noqa: F401
    EQUIVALENCES, PONCT_CLE, PONCTUATION_ARABE, TABLE_PONCT, TABLE_VOCAL,
    VOCALISATION, cle_arabe, cle_latine, normaliser_cible)


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
