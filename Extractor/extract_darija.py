import pandas as pd
import requests

SOURCE = "DrIAmed/darija-youtube-dataset"
PARQUET = f"hf://datasets/{SOURCE}/data/train-00000-of-00001.parquet"
OUTPUT = "darija_translit.csv"
COLS = ["darija_Arab_new", "darija_Latn"]


def load_via_parquet():
    # Lecture à distance des 2 colonnes seulement (pas de téléchargement du fichier complet)
    return pd.read_parquet(PARQUET, columns=COLS)


def load_via_api():
    # Plan B : API publique de Hugging Face, 100 lignes par requête
    url = "https://datasets-server.huggingface.co/rows"
    rows, offset = [], 0
    while True:
        r = requests.get(url, params={
            "dataset": SOURCE, "config": "default", "split": "train",
            "offset": offset, "length": 100,
        }, timeout=60)
        r.raise_for_status()
        data = r.json()
        batch = [{c: x["row"][c] for c in COLS} for x in data["rows"]]
        rows.extend(batch)
        offset += 100
        if offset >= data["num_rows_total"]:
            break
    return pd.DataFrame(rows)


try:
    df = load_via_parquet()
except Exception as e:
    print(f"Lecture Parquet impossible ({e}), passage par l'API...")
    df = load_via_api()

df = df.rename(columns={
    "darija_Arab_new": "darija",
    "darija_Latn": "transliteration",
})
df.insert(0, "id", [f"darija-youtube_{i:05d}" for i in range(len(df))])
df["source"] = SOURCE

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"{len(df)} lignes enregistrées dans {OUTPUT}")