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
| `merge_all.py` | `darija_all.csv` |

`atlasia/DODa-audio-dataset-V3` n'a pas encore de script : son CSV se récupère
depuis le dépôt Hugging Face. `merge_all.py` signale les fichiers absents et
poursuit avec les autres.


### Gestion des ambiguïtés et normalisation par LLM

Après le nettoyage initial, une problématique majeure a persisté et complexifié le projet : la gestion des entrées en Arabizi ayant plusieurs sorties possibles en Darija.

Pour analyser l'ampleur de ce problème, nous avons développé le script de détection dans `Extractor/conflict_detector.py`. Les résultats obtenus sont les suivants :
- **Total de lignes en conflit :** 21 711
- **Nombre de formes Arabizi ambiguës :** 9 546
- **Fichier exporté avec succès :** `arabizi_vers_darija_conflits.xlsx`

Face à ce volume élevé de formes ambiguës et au temps restreint dont nous disposons, une validation humaine de chaque cas s'est avérée impossible. La stratégie adoptée consiste donc à exploiter de **grands modèles de langage (LLM)** pour effectuer directement la normalisation, s'affranchissant ainsi d'une validation manuelle fastidieuse.



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
```

## Problèmes courants

`Unable to create process using ...pip-script.py`
: Chemin utilisateur contenant un espace sous Windows. Utiliser
  `python -m pip install ...`.

`Espace insuffisant sur le disque`
: Ne pas utiliser `datasets.load_dataset` pour `darija-youtube-dataset`
  (téléchargement complet avec l'audio). Utiliser le script fourni, qui lit
  seulement les colonnes de texte.
