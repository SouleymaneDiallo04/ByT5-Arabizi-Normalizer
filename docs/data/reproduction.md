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
| `Extractor/extract_doda_audio.py` | `doda_audio_translit.csv` |
| `merge_all.py` | `darija_all.csv` |
| `make_splits.py` | `train.csv`, `dev.csv`, `test_real.csv` |

Les six scripts couvrent désormais toutes les sources : le pipeline est
intégralement reproductible, ce qu'exige ENF-04. `merge_all.py` signale les
fichiers absents et poursuit avec les autres.

`extract_doda_audio.py` demande d'avoir accepté une fois les conditions du
dépôt sur sa page, puis d'être authentifié — `hf auth login`, ou la variable
`HF_TOKEN`. L'accord est automatique.

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

python merge_all.py
python make_splits.py
```

## Problèmes courants

`Unable to create process using ...pip-script.py`
: Chemin utilisateur contenant un espace sous Windows. Utiliser
  `python -m pip install ...`.

`Espace insuffisant sur le disque`
: Ne pas utiliser `datasets.load_dataset` pour `darija-youtube-dataset`
  (téléchargement complet avec l'audio). Utiliser le script fourni, qui lit
  seulement les colonnes de texte.
