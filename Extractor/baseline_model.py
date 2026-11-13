import pandas as pd
import re
from collections import Counter

# 1. Charger le corpus nettoyé
df = pd.read_csv("darija_all.csv")

# 2. Construction automatique du dictionnaire (Lexique de référence)
# On associe chaque Arabizi (key_la) à sa cible arabe (darija) la plus fréquente
lexique_frequences = {}

for _, row in df.dropna(subset=["darija", "transliteration"]).iterrows():
    mots_la = str(row["transliteration"]).split()
    mots_ar = str(row["darija"]).split()
    
    # Si les phrases ont la même longueur de mots, on peut faire un alignement simple mot à mot
    if len(mots_la) == len(mots_ar):
        for la, ar in zip(mots_la, mots_ar):
            la_clean = la.lower().strip()
            if la_clean not in lexique_frequences:
                lexique_frequences[la_clean] = Counter()
            lexique_frequences[la_clean][ar] += 1

# On garde pour chaque mot Arabizi la traduction arabe la plus fréquente
dictionnaire_baseline = {
    mot: counts.most_common(1)[0][0] for mot, counts in lexique_frequences.items()
}

print(f"Dictionnaire de baseline généré : {len(dictionnaire_baseline)} entrées.")

# 3. Règles de secours (Phonétique Arabizi -> Arabe classique/dialectal)
PHONETIC_FALLBACK = {
    '3': 'ع',
    '7': 'ح',
    '5': 'خ',
    '9': 'ق',
    '2': 'ء',
    '8': 'غ',
    '6': 'ط'
}

def normaliser_baseline(texte_arabizi):
    """
    Normalise une phrase en arabizi en utilisant le dictionnaire 
    et les règles phonétiques de secours pour les mots inconnus.
    """
    mots = str(texte_arabizi).split()
    mots_normalises = []
    
    for mot in mots:
        mot_clean = mot.lower().strip()
        # 1. Recherche dans le dictionnaire exact
        if mot_clean in dictionnaire_baseline:
            mots_normalises.append(dictionnaire_baseline[mot_clean])
        else:
            # 2. Fallback phonétique basique si le mot est inconnu
            mot_transcrit = mot
            for chiffre, lettre in PHONETIC_FALLBACK.items():
                mot_transcrit = mot_transcrit.replace(chiffre, lettre)
            mots_normalises.append(mot_transcrit)
            
    return " ".join(mots_normalises)

# Test rapide sur un exemple
exemple = "salam 3likom 3lach madertch lkhdma"
print("Entrée Arabizi :", exemple)
print("Sortie Baseline :", normaliser_baseline(exemple))