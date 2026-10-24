# Démarrage rapide

Reconstruire le corpus complet en quelques commandes, depuis la racine du dépôt :

```bash
python extract_darija.py
python extract_doda.py
python extract_daa_pairs.py
python extract_doda_synthetic.py
python merge_all.py
```

Chaque script de la première série produit un fichier CSV (par exemple
`doda_translit.csv`) et affiche le nombre de lignes enregistrées.
`merge_all.py` produit `darija_all.csv`.

Les scripts lisent seulement les colonnes utiles des fichiers Parquet publiés
sur Hugging Face : il n'est pas nécessaire de télécharger les jeux de données
en entier. Voir {doc}`../data/reproduction` pour le détail.

## Vérifier le résultat

```python
import pandas as pd

df = pd.read_csv("darija_all.csv")
print(df.shape)
print(df["source"].value_counts())
```
