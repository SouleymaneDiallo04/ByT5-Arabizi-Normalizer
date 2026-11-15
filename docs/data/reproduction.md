# Reproduire le jeu de données

Tout le code est dans le dépôt GitHub :
<https://github.com/SouleymaneDiallo04/ByT5-Arabizi-Normalizer>

## Scripts

| Script | Produit |
|---|---|
| `donnees/extraction/extract_darija.py` | `darija_translit.csv` |
| `donnees/extraction/extract_doda.py` | `doda_translit.csv` |
| `donnees/extraction/extract_daa_pairs.py` | `daa_pairs_translit.csv` |
| `donnees/extraction/extract_doda_synthetic.py` | `doda_synthetic_translit.csv` |
| `donnees/extraction/extract_doda_audio.py` | `doda_audio_translit.csv` |
| `donnees/merge_all.py` | `darija_all.csv` |
| `donnees/make_splits.py` | `train.csv`, `dev.csv`, `test_real.csv` |

Les six scripts couvrent désormais toutes les sources : le pipeline est
intégralement reproductible, ce qu'exige ENF-04. `merge_all.py` signale les
fichiers absents et poursuit avec les autres.

`extract_doda_audio.py` demande d'avoir accepté une fois les conditions du
dépôt sur sa page, puis d'être authentifié — `hf auth login`, ou la variable
`HF_TOKEN`. L'accord est automatique.


## Lignes en conflit : export pour relecture

Après le nettoyage, une difficulté demeure : un même arabizi peut recevoir
plusieurs cibles arabes, selon la source. `merge_all.py` marque ces lignes du
drapeau `conflit` sans les supprimer (voir {doc}`limitations`). Deux scripts
les sortent du corpus pour qu'un locuteur les arbitre :

| Script | Produit |
|---|---|
| `donnees/conflict_detector.py` | `arabizi_vers_darija_conflits.xlsx`, avec deux colonnes à remplir : `action_validation` et `darija_valide` |
| `donnees/split_file.py` | le même fichier découpé en six parts, `arabizi_conflits_partie_1.xlsx` à `_6.xlsx`, pour répartir la relecture |

```bash
python -m donnees.conflict_detector
python -m donnees.split_file
```

À son écriture, le premier script relevait **21 711 lignes en conflit, pour
9 546 formes arabizi ambiguës** ; ces chiffres suivent le corpus. Le découpage
se fait sur la forme arabizi normalisée, `key_la` : toutes les cibles
concurrentes d'une même forme arrivent dans la même part.

```{note}
Valider à la main 9 546 formes n'est pas tenable dans le temps du projet. Une
piste proposée est de confier cette normalisation à un grand modèle de langage.
Le pipeline actuel ne l'applique pas : les lignes en conflit sont conservées
telles quelles. L'adopter changerait les cibles d'entraînement, et la décision
devra passer par le {doc}`journal des décisions <../project/decisions>`.
```

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
python -m donnees.extraction.extract_darija
python -m donnees.extraction.extract_doda
python -m donnees.extraction.extract_daa_pairs
python -m donnees.extraction.extract_doda_synthetic
python -m donnees.extraction.extract_doda_audio

python -m donnees.merge_all
python -m donnees.make_splits
```

## Problèmes courants

`ModuleNotFoundError: No module named 'evaluation'` (ou `'modele'`, `'donnees'`)
: Le script a été lancé par son chemin, `python donnees/merge_all.py`, ou
  depuis un autre répertoire que la racine. Les scripts s'importent d'un dossier
  à l'autre : il faut les lancer en module, depuis la racine du dépôt,
  `python -m donnees.merge_all`.

`Unable to create process using ...pip-script.py`
: Chemin utilisateur contenant un espace sous Windows. Utiliser
  `python -m pip install ...`.

`Espace insuffisant sur le disque`
: Ne pas utiliser `datasets.load_dataset` pour `darija-youtube-dataset`
  (téléchargement complet avec l'audio). Utiliser le script fourni, qui lit
  seulement les colonnes de texte.
