import pandas as pd
import requests

SOURCE = "MBZUAI-Paris/DODa-10K"
PARQUET = f"hf://datasets/{SOURCE}@refs/convert/parquet/default/train/0000.parquet"
OUTPUT = "doda_translit.csv"
COLS = ["index", "Darija_arabic", "Darija_latin"]


def load_via_parquet():
    # Lecture à distance des 3 colonnes seulement
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
        rows.extend({c: x["row"][c] for c in COLS} for x in data["rows"])
        offset += 100
        if offset >= data["num_rows_total"]:
            break
    return pd.DataFrame(rows)


try:
    df = load_via_parquet()
except Exception as e:
    print(f"Lecture Parquet impossible ({e}), passage par l'API...")
    df = load_via_api()

out = pd.DataFrame({
    "id": [f"doda10k_{int(i):05d}" for i in df["index"]],
    "darija": df["Darija_arabic"],
    "transliteration": df["Darija_latin"],
    "source": SOURCE,
})

out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"{len(out)} lignes enregistrées dans {OUTPUT}")