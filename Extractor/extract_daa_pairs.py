import pandas as pd
import requests

SOURCE = "OiQ/daa-pairs"
PARQUET = f"hf://datasets/{SOURCE}@refs/convert/parquet/default/train/0000.parquet"
OUTPUT = "daa_pairs_translit.csv"
COLS = ["arabic", "arabizi"]


def load_via_parquet():
    # Lecture à distance des 2 colonnes seulement
    return pd.read_parquet(PARQUET, columns=COLS)


def load_via_parquet_list():
    # Plan B : récupère la liste des fichiers Parquet via l'API, puis lit les 2 colonnes
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
    "id": [f"daa-pairs_{i:06d}" for i in range(len(df))],
    "darija": df["arabic"],
    "transliteration": df["arabizi"],
    "source": SOURCE,
})

out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"{len(out)} lignes enregistrées dans {OUTPUT}")