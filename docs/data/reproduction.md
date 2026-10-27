# Reproduire le jeu de données

Tout le code est dans le dépôt GitHub :
<https://github.com/SouleymaneDiallo04/ByT5-Arabizi-Normalizer>

## Scripts

| Script | Produit |
|---|---|
| `Extractor/extract_darija.py` | `darija_translit.csv` |
| `Extractor/extract_doda.py` | `doda_translit.csv` |
| `Extractor/extract_daa_pairs.py` | `daa_pairs_translit.csv` |
| `Extractor/extract_doda_synthetic.py` | `doda_synthetic_translit.csv` |

`merge_all.py`, qui produira `darija_all.csv` (fusion, nettoyage,
dédoublonnage), relève de la phase suivante et n'est pas encore écrit.

## Principe

Les scripts lisent directement, à distance, les colonnes utiles des fichiers
Parquet publiés sur Hugging Face. Ils ne téléchargent ni l'audio ni les colonnes
inutiles. C'est ce qui permet de les exécuter même avec peu d'espace disque :
`DrIAmed/darija-youtube-dataset` pèse environ 475 Mo avec l'audio, alors que
seules deux colonnes de texte sont nécessaires.

Chaque script tente d'abord la lecture Parquet directe, puis bascule sur une
API publique de Hugging Face si elle échoue.

## Commandes

```bash
python Extractor/extract_darija.py
python Extractor/extract_doda.py
python Extractor/extract_daa_pairs.py
python Extractor/extract_doda_synthetic.py
```

## Problèmes courants

`Unable to create process using ...pip-script.py`
: Chemin utilisateur contenant un espace sous Windows. Utiliser
  `python -m pip install ...`.

`Espace insuffisant sur le disque`
: Ne pas utiliser `datasets.load_dataset` pour `darija-youtube-dataset`
  (téléchargement complet avec l'audio). Utiliser le script fourni, qui lit
  seulement les colonnes de texte.
