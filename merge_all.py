import pandas as pd
import re
import unicodedata

INPUTS = [
    "doda_translit.csv",
    "darija_translit.csv",
    "daa_pairs_translit.csv",
    "doda_synthetic_translit.csv",
    "doda_audio_v3_translit.csv",
]
OUTPUT = "darija_all.csv"
COLS = ["id", "darija", "transliteration", "source"]

# Ponctuation neutralisée dans les clés : ASCII, arabe (، ؛ ؟ ۔) et typographique
PONCTUATION = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "،؛؟٪٫٬٭۔"
    "«»‘’“”…–—"
)
# Signes de vocalisation (shadda, fatha...) et tatweel, retirés des clés
VOCALISATION = [chr(c) for c in range(0x064B, 0x0656)] + ["ـ", "ٰ"]
# Variantes de lettres unifiées dans les clés
EQUIVALENCES = [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")]

# Consigne laissée dans la cible par le modèle qui a généré la source. Le motif est
# volontairement étroit : les mots français de la cible sont du code-switching voulu.
ARTEFACTS = re.compile(r"(?i)\btranslation should be\b")
# Bande plausible du rapport longueur(arabe)/longueur(arabizi), mesurée sur le corpus
RATIO_MIN, RATIO_MAX = 0.45, 1.35

TABLE_PONCT = {ord(c): " " for c in PONCTUATION}
TABLE_VOCAL = {ord(c): None for c in VOCALISATION}


def nettoyer(texte):
    # Normalise les espaces : les sous-titres YouTube contiennent des retours à la ligne
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(texte))).strip()


def cle_arabe(texte):
    # Clé de comparaison seulement : ponctuation, vocalisation et variantes neutralisées
    t = texte.translate(TABLE_PONCT).translate(TABLE_VOCAL)
    for avant, apres in EQUIVALENCES:
        t = t.replace(avant, apres)
    return re.sub(r"\s+", " ", t).strip()


def cle_latine(texte):
    return re.sub(r"\s+", " ", texte.lower().translate(TABLE_PONCT)).strip()


def richesse(texte):
    # Départage les doublons : on garde la copie vocalisée, plus informative
    return sum(ord(c) in TABLE_VOCAL for c in texte)


frames = []
for nom in INPUTS:
    try:
        frames.append(pd.read_csv(nom, usecols=COLS))
    except FileNotFoundError:
        print(f"{nom} absent, ignoré (lancer son script dans Extractor/)")

if not frames:
    raise SystemExit("Aucun fichier d'entrée : lancer d'abord les scripts Extractor/")

df = pd.concat(frames, ignore_index=True)
depart = len(df)

# 1. Lignes inutilisables, et espaces normalisés
df = df.dropna(subset=["darija", "transliteration"])
df["darija"] = df["darija"].map(nettoyer)
df["transliteration"] = df["transliteration"].map(nettoyer)
df = df[(df["darija"] != "") & (df["transliteration"] != "")]
vides = depart - len(df)

# 2. Consignes de génération restées dans la cible
avant = len(df)
df = df[~df["darija"].str.contains(ARTEFACTS, na=False)]
artefacts = avant - len(df)

# 3. Clés de comparaison. Le texte d'origine n'est pas modifié au-delà des espaces :
#    c'est lui qui servira à l'entraînement, pas les clés.
df["key_ar"] = df["darija"].map(cle_arabe)
df["key_la"] = df["transliteration"].map(cle_latine)

# 4. Doublons de paires : même phrase et même arabizi aux variantes d'écriture près.
avant = len(df)
df = (
    df.assign(_richesse=df["darija"].map(richesse))
    .sort_values("_richesse", ascending=False, kind="stable")
    .drop_duplicates(subset=["key_ar", "key_la"], keep="first")
    .drop(columns="_richesse")
    .sort_index()
)
doublons = avant - len(df)

# 5. Même phrase arabe écrite de plusieurs façons en arabizi : conservé tel quel,
#    c'est la variation d'écriture que le modèle doit apprendre à absorber.
variantes = int((df.groupby("key_ar")["key_la"].transform("nunique") > 1).sum())

# 6. Même arabizi mais cibles arabes divergentes : marqué et non supprimé,
#    l'arbitrage dépend de la convention orthographique encore à fixer.
df["conflit"] = df.groupby("key_la")["key_ar"].transform("nunique") > 1

# 7. Paires dont les deux côtés ne se correspondent manifestement pas. Marqué plutôt
#    que supprimé : certains écarts sont des chiffres écrits en mots (3ndi 32).
ratio = df["key_ar"].str.len() / df["key_la"].str.len().clip(lower=1)
df["suspect"] = (ratio < RATIO_MIN) | (ratio > RATIO_MAX)

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"{depart} lignes lues dans {len(frames)} fichier(s)")
print(f"  {vides} lignes vides ou inutilisables retirées")
print(f"  {artefacts} consignes de génération retirées")
print(f"  {doublons} doublons de paires retirés")
print(f"  {variantes} lignes de variation arabizi conservées")
print(f"  {int(df['conflit'].sum())} lignes marquées conflit (cibles divergentes)")
print(f"  {int(df['suspect'].sum())} lignes marquées suspect (alignement douteux)")
print(f"{len(df)} lignes enregistrées dans {OUTPUT}")
print()
print(df["source"].value_counts().to_string())
