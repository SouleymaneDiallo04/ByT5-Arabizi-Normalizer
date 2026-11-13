import numpy as np
import pandas as pd

# 1. Chargement du fichier source
INPUT_EXCEL = "arabizi_vers_darija_conflits.xlsx"
df = pd.read_excel(INPUT_EXCEL)

# 2. Extraction des mots uniques pour conserver les groupes intacts
mots_uniques = df["transliteration"].unique()

# 3. Division de la liste des mots uniques en 6 parties égales
chunks_mots = np.array_split(mots_uniques, 6)

# 4. Filtrage et enregistrement des 6 fichiers Excel
for i, chunk in enumerate(chunks_mots, start=1):
    df_partie = df[df["transliteration"].isin(chunk)]
    nom_fichier = f"arabizi_conflits_partie_{i}.xlsx"
    df_partie.to_excel(nom_fichier, index=False, engine="openpyxl")

    print(f"Partie {i} créée : {nom_fichier}")
    print(f"  - Mots uniques : {len(chunk)}")
    print(f"  - Lignes totales : {len(df_partie)}\n")