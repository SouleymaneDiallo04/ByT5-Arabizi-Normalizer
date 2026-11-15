"""Extraction de atlasia/DODa-audio-dataset.

Ce jeu etait jusqu'ici la seule source du corpus sans script : son CSV se
recuperait a la main depuis Hugging Face, ce qui laissait un trou dans la
reproductibilite exigee par ENF-04. Ce script le comble.

Il vise \x60atlasia/DODa-audio-dataset\x60 et non la variante V3 employee jusqu'ici.
Mesure : 12 721 paires exploitables contre 8 697 conservees de la V3 apres
nettoyage, dont 3 668 absentes du corpus. La V3 y est contenue.

Le jeu porte deux transcriptions arabes de chaque phrase. \x60darija_Arab_old\x60 est
une transcription phonetique, vocalisee et ponctuee ; \x60darija_Arab_new\x60 suit une
orthographe conventionnelle, sans vocalisation. C'est la seconde qui est
retenue, conformement a PO-01.

La paire des deux colonnes constitue par ailleurs une ressource a part entiere :
11 391 phrases attestees sous deux orthographes par les memes annotateurs, soit
une table de normalisation que le projet n'avait pas. Elle est conservee dans le
fichier de sortie, meme si merge_all.py ne l'utilise pas.

La colonne audio n'est jamais lue : elle pese 1,29 Go pour rien, seul le texte
nous interesse.

Le depot est en acces automatique : il faut avoir accepte ses conditions une
fois sur sa page, puis etre authentifie (hf auth login, ou la variable
HF_TOKEN).
"""

import pandas as pd
import requests
from huggingface_hub import get_token

SOURCE = "atlasia/DODa-audio-dataset"
OUTPUT = "doda_audio_translit.csv"

COL_TRANSLIT = "darija_Latn"        # arabizi
COL_DARIJA = "darija_Arab_new"      # cible, orthographe conventionnelle
COL_PHONETIQUE = "darija_Arab_old"  # meme phrase, transcription phonetique
COLS = [COL_TRANSLIT, COL_DARIJA, COL_PHONETIQUE]


def entetes():
    jeton = get_token()
    if not jeton:
        raise SystemExit(
            "Aucun jeton Hugging Face. Ce depot demande d'avoir accepte ses "
            "conditions sur sa page, puis : hf auth login")
    return {"Authorization": f"Bearer {jeton}"}


def charger():
    """Liste les fichiers Parquet puis ne lit que les trois colonnes utiles."""
    h = entetes()
    r = requests.get("https://datasets-server.huggingface.co/parquet",
                     params={"dataset": SOURCE}, headers=h, timeout=90)
    r.raise_for_status()
    urls = [f["url"] for f in r.json()["parquet_files"] if f["split"] == "train"]
    return pd.concat([pd.read_parquet(u, columns=COLS, storage_options=h)
                      for u in urls], ignore_index=True)


df = charger()
avant = len(df)
df = df.dropna(subset=[COL_TRANSLIT, COL_DARIJA])
df = df[(df[COL_TRANSLIT].astype(str).str.strip() != "")
        & (df[COL_DARIJA].astype(str).str.strip() != "")]

out = pd.DataFrame({
    "id": [f"doda-audio_{i:05d}" for i in range(len(df))],
    "darija": df[COL_DARIJA].values,
    "transliteration": df[COL_TRANSLIT].values,
    "source": SOURCE,
    "darija_phonetique": df[COL_PHONETIQUE].values,
})
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"{avant} lignes lues, {len(out)} conservees dans {OUTPUT}")
print(f"{avant - len(out)} ecartees pour un cote vide")
divergent = int((out["darija"].astype(str) != out["darija_phonetique"].astype(str)).sum())
print(f"{divergent} phrases portent deux orthographes distinctes "
      f"({100 * divergent / max(len(out), 1):.1f} %)")
