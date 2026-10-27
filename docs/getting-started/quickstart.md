# Démarrage rapide

Reconstruire le corpus complet en quelques commandes, depuis la racine du dépôt :

```bash
python Extractor/extract_darija.py
python Extractor/extract_doda.py
python Extractor/extract_daa_pairs.py
python Extractor/extract_doda_synthetic.py
```

Chaque script produit un fichier CSV (par exemple `doda_translit.csv`) et
affiche le nombre de lignes enregistrées. La fusion des quatre fichiers en
`darija_all.csv` relève de la phase de nettoyage et n'est pas encore écrite.

Les scripts lisent seulement les colonnes utiles des fichiers Parquet publiés
sur Hugging Face : il n'est pas nécessaire de télécharger les jeux de données
en entier. Voir {doc}`../data/reproduction` pour le détail.

## Vérifier le résultat

```python
import pandas as pd

df = pd.read_csv("doda_translit.csv")
print(df.shape)
print(df["source"].value_counts())
```
