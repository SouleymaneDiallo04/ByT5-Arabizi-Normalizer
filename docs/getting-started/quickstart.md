# Démarrage rapide

Reconstruire le corpus complet en quelques commandes, depuis la racine du dépôt :

```bash
python -m donnees.extraction.extract_darija
python -m donnees.extraction.extract_doda
python -m donnees.extraction.extract_daa_pairs
python -m donnees.extraction.extract_doda_synthetic
python -m donnees.extraction.extract_doda_audio

python -m donnees.merge_all
python -m donnees.make_splits
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
