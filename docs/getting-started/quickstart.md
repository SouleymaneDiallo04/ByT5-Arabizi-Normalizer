# Démarrage rapide

Reconstruire le corpus complet en quelques commandes, depuis la racine du dépôt :

```bash
python Extractor/extract_darija.py
python Extractor/extract_doda.py
python Extractor/extract_daa_pairs.py
python Extractor/extract_doda_synthetic.py

python merge_all.py
python make_splits.py
```

Chaque script d'extraction produit un fichier CSV (par exemple
`doda_translit.csv`) et affiche le nombre de lignes enregistrées. `merge_all.py`
les empile ensuite en `darija_all.csv` et détaille ce qu'il a retiré et marqué.
`make_splits.py` en tire enfin `train.csv`, `dev.csv` et `test_real.csv`.

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
