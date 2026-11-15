import re
import difflib
from collections import Counter, defaultdict

import pandas as pd
import requests

SOURCE = "OiQ/daa-pairs"
PARQUET = f"hf://datasets/{SOURCE}@refs/convert/parquet/default/train/0000.parquet"
OUTPUT = "seed_lexicon_from_daa.xlsx"
MIN_COUNT = 2
COLS = ["arabic", "mixte"]


def load_via_parquet():
    return pd.read_parquet(PARQUET, columns=COLS)


def load_via_parquet_list():
    r = requests.get("https://datasets-server.huggingface.co/parquet",
                     params={"dataset": SOURCE}, timeout=60)
    r.raise_for_status()
    urls = [f["url"] for f in r.json()["parquet_files"] if f["split"] == "train"]
    return pd.concat([pd.read_parquet(u, columns=COLS) for u in urls], ignore_index=True)


try:
    df = load_via_parquet()
except Exception as e:
    print(f"Lecture directe impossible ({e}), passage par la liste des fichiers...")
    df = load_via_parquet_list()

PUNCT = "،,.!?؟:;\"«»()[]…-"
LATIN = re.compile(r"[A-Za-z]")


def tokens(text):
    out = []
    for t in str(text).split():
        t = t.strip(PUNCT)
        if t:
            out.append(t)
    return out


pairs = Counter()
for ar, mx in zip(df["arabic"], df["mixte"]):
    a, m = tokens(ar), tokens(mx)
    sm = difflib.SequenceMatcher(None, m, a, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        # On ne garde que les remplacements 1 mot <-> 1 mot (alignement fiable)
        if op == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
            lat, arb = m[i1], a[j1]
            if LATIN.search(lat) and not LATIN.search(arb):
                pairs[(lat.lower(), arb)] += 1

by_latin = defaultdict(int)
for (lat, _), c in pairs.items():
    by_latin[lat] += c

rows = [
    {"latin": lat, "arabic": arb, "count": c, "share": round(c / by_latin[lat], 3)}
    for (lat, arb), c in pairs.items() if c >= MIN_COUNT
]
out = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

# Écriture Excel avec mise en forme de base
with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name="lexicon")
    ws = writer.sheets["lexicon"]
    ws.freeze_panes = "A2"                 # en-tête figé
    ws.auto_filter.ref = ws.dimensions     # filtres sur toutes les colonnes
    for col, width in zip("ABCD", (28, 28, 10, 10)):
        ws.column_dimensions[col].width = width

print(f"{len(out)} paires (latin -> arabe) enregistrées dans {OUTPUT}")
print(out.head(20).to_string(index=False))