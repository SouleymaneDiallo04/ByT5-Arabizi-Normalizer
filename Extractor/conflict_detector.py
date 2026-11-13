import pandas as pd

INPUT_CSV = "darija_all.csv"
OUTPUT_EXCEL = "arabizi_vers_darija_conflits.xlsx"

# 1. Charger le corpus
df = pd.read_csv(INPUT_CSV)

# 2. Identifier les clés Arabizi (key_la) qui ont plus d'une translittération en Darija (key_ar)
groupes_conflits = df.groupby("key_la")["key_ar"].transform("nunique") > 1
df_conflits = df[groupes_conflits].copy()

# 3. Trier par la clé Arabizi pour avoir toutes les variantes d'une même phrase l'une sous l'autre
df_conflits = df_conflits.sort_values(by=["key_la", "source"])

# 4. Ajouter des colonnes pour la validation humaine
df_conflits["action_validation"] = ""  # Ex: "GARDER", "SUPPRIMER", "CORRIGER"
df_conflits["darija_valide"] = ""     # La forme exacte en Darija arabe validée par le locuteur

# 5. Sélection des colonnes pour la relecture
cols = [
    "id",
    "source",
    "transliteration",  # Arabizi d'origine
    "darija",           # Translittération en Darija arabe proposée
    "key_la",
    "action_validation",
    "darija_valide"
]

# 6. Export vers Excel
df_conflits[cols].to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")

print(f"Total de lignes en conflit : {len(df_conflits)}")
print(f"Nombre de formes Arabizi ambiguës : {df_conflits['key_la'].nunique()}")
print(f"Fichier exporté avec succès : {OUTPUT_EXCEL}")