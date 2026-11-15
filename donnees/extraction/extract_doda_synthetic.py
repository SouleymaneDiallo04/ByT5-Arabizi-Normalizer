import pandas as pd
import requests

SOURCE = "Haitam03/doda_synthetic"
PARQUET = f"hf://datasets/{SOURCE}@refs/convert/parquet/default/train/0000.parquet"
OUTPUT = "doda_synthetic_translit.csv"

COL_DARIJA = "corrected_output"   # darija en écriture arabe (corrigé)
COL_TRANSLIT = "latin_input"      # arabizi brut
# Pour l'arabizi normalisé, remplace COL_TRANSLIT par "normalized_latin_darija"
COLS = [COL_DARIJA, COL_TRANSLIT]


def load_via_parquet():
    # Lecture à distance des 2 colonnes seulement
    return pd.read_parquet(PARQUET, columns=COLS)


def load_via_parquet_list():
    # Plan B : liste des fichiers Parquet via l'API, puis lecture des 2 colonnes
    r = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": SOURCE}, timeout=60,
    )
    r.raise_for_status()
    urls = [f["url"] for f in r.json()["parquet_files"] if f["split"] == "train"]
    return pd.concat(
        [pd.read_parquet(u, columns=COLS) for u in urls], ignore_index=True
    )


try:
    df = load_via_parquet()
except Exception as e:
    print(f"Lecture directe impossible ({e}), passage par la liste des fichiers...")
    df = load_via_parquet_list()

out = pd.DataFrame({
    "id": [f"doda-synthetic_{i:05d}" for i in range(len(df))],
    "darija": df[COL_DARIJA],
    "transliteration": df[COL_TRANSLIT],
    "source": SOURCE,
})

out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"{len(out)} lignes enregistrées dans {OUTPUT}")