# ByT5-Arabizi-Normalizer

*Normalisation de l'arabizi marocain vers l'écriture arabe avec ByT5*

[![Documentation](https://readthedocs.org/projects/byt5-arabizi-normalizer/badge/?version=latest)](https://byt5-arabizi-normalizer.readthedocs.io/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Statut](https://img.shields.io/badge/statut-en%20d%C3%A9veloppement-orange)

## Introduction

<p align="center">
  <img src="docs/_static/images/Podcast_Maroc_im.png" alt="Illustration du projet" width="100%">
</p>

### Contexte

Le darija est la langue de communication quotidienne au Maroc, mais il est peu écrit de façon formelle. Sur les réseaux sociaux et les messageries, il est très souvent écrit en **arabizi** : des lettres latines complétées par des chiffres qui représentent des sons absents du latin (`3` pour ع, `7` pour ح, `9` pour ق). Cette écriture n'a pas de norme : chacun écrit comme il prononce, selon ses habitudes et son clavier.

Or la plupart des outils de traitement automatique de l'arabe attendent de l'écriture arabe, et les ressources disponibles pour le darija sont bien plus rares que pour l'arabe standard ou le français. Tant que ces messages restent en arabizi, ils sont difficiles à exploiter.

### Problématique

> **Comment convertir de façon fiable un message en arabizi marocain, bruité et non standardisé, y compris lorsqu'il mêle du français, vers du darija écrit en caractères arabes, alors que les données parallèles disponibles sont peu nombreuses, hétérogènes et en grande partie synthétiques ?**

Quatre difficultés rendent la tâche non triviale :

- **Une écriture sans norme.** Un même mot s'écrit de plusieurs façons (`merci`, `mersi`, `mrc`), et un même caractère latin peut correspondre à plusieurs lettres arabes : `ga3` s'écrit ڭاع alors que `ghanmchiw` s'écrit غانمشيو.
- **Une cible elle aussi variable.** L'orthographe du darija en caractères arabes n'est pas standardisée : dans `Haitam03/doda_synthetic`, `ngoulik` apparaît sous la forme نڭوليك dans la sortie brute d'un modèle et نقوليك dans la version corrigée. Il faut fixer une convention.
- **Le mélange avec le français.** Les locuteurs insèrent des mots et expressions françaises (`l'adresse`, `stp`, `application`) qu'un modèle ne connaissant pas ces cas risque de déformer.
- **Des données limitées et inégales.** Les sources existantes ont des conventions différentes, plus de 90 % des lignes collectées sont synthétiques, et il n'existe pas de jeu de test réel.

### Ce que nous construisons

**Un corpus de paires darija arabe ↔ arabizi**, rassemblé à partir de sources publiques puis enrichi :

- un **lexique de mots français fréquents**, par domaine (santé, technologie, administration, banque, école, commerce, etc.), avec leurs variantes d'écriture latine et leur forme en écriture arabe ;
- des **phrases mixtes darija / français** générées à partir de ce lexique, avec la version en écriture arabe correspondante, et du bruit réaliste sur l'arabizi (apostrophes oubliées, voyelles supprimées, chiffres variables) ;
- un **Nouveau Testament en darija**, traduit depuis le français (texte Louis Segond 1910, dans le domaine public) par un modèle de langage, puis validé par un second modèle ;
- un **jeu de test réel**, relu à la main et jamais utilisé pour générer des données d'entraînement.

**Un modèle de normalisation** basé sur **ByT5**, utilisé en séquence-à-séquence : il reçoit une phrase en arabizi et produit la phrase correspondante en darija écrit en caractères arabes. ByT5 traite le texte octet par octet, sans vocabulaire fermé : une faute de frappe, une variante d'orthographe ou un mot français jamais vu ne sont pas « hors vocabulaire ». Il sera évalué sur le jeu de test réel (taux d'erreur de caractères, correspondance exacte), avec une analyse d'erreurs par source et par type de phrase.

Le projet normalise l'écriture ; il ne cherche pas à traduire le darija, ni à identifier le dialecte.

> **Statut.** Les cinq sources sont extraites, unifiées et fusionnées en un corpus nettoyé. Le lexique, les phrases mixtes, le jeu de test et l'entraînement du modèle restent à réaliser.

## Données

| Source | Lignes | Nature |
|---|---|---|
| [`DrIAmed/darija-youtube-dataset`](https://huggingface.co/datasets/DrIAmed/darija-youtube-dataset) | 2 706 | Transcriptions de vidéos YouTube |
| [`MBZUAI-Paris/DODa-10K`](https://huggingface.co/datasets/MBZUAI-Paris/DODa-10K) | 10 000 | Corpus de référence |
| [`OiQ/daa-pairs`](https://huggingface.co/datasets/OiQ/daa-pairs) | 112 814 | Synthétique (généré par LLM) |
| [`Haitam03/doda_synthetic`](https://huggingface.co/datasets/Haitam03/doda_synthetic) | 64 260 | Synthétique, arabizi bruité |
| [`atlasia/DODa-audio-dataset-V3`](https://huggingface.co/datasets/atlasia/DODa-audio-dataset-V3) | 11 517 | Transcriptions d'enregistrements |
| **Total (avant nettoyage)** | **201 297** | |
| **Après fusion et nettoyage** | **196 770** | |

Chaque fichier produit suit le même schéma :

| Colonne | Contenu |
|---|---|
| `id` | Identifiant unique, préfixé selon la source |
| `darija` | Phrase en écriture arabe |
| `transliteration` | La même phrase en arabizi |
| `source` | Jeu de données d'origine |

Le corpus est publié sur Hugging Face : <https://huggingface.co/datasets/sawti-nlp/darija-arabizi-corpus>
(dépôt privé pour l'instant ; l'accès doit être accordé aux encadrants, ou le dépôt passé en public.)

## Structure du dépôt

Les scripts sont rangés selon l'étape qu'ils servent : construire le corpus,
entraîner, mesurer.

```
.
├── donnees/                        # construire le corpus
│   ├── extraction/                 # une source publique par script
│   │   ├── extract_darija.py           # DrIAmed/darija-youtube-dataset
│   │   ├── extract_doda.py             # MBZUAI-Paris/DODa-10K
│   │   ├── extract_daa_pairs.py        # OiQ/daa-pairs
│   │   ├── extract_doda_synthetic.py   # Haitam03/doda_synthetic
│   │   ├── extract_doda_audio.py       # atlasia/DODa-audio-dataset
│   │   ├── extract_nilechat.py         # UBC-NLP/nilechat, joint à sa source arabe
│   │   └── seed_lexicon.py             # amorce du lexique de mots français
│   ├── merge_all.py                # fusion, nettoyage, dédoublonnage
│   ├── make_splits.py              # jeu de test et découpage train / dev
│   ├── decouper_test.py            # jeu de réglage prélevé dans le test
│   ├── augmenter.py                # augmentation par graphies attestées
│   ├── ajouter_source.py           # nouvelle source au train, dev et test intacts
│   ├── melanger.py                 # mélange avec un corpus candidat, à budget fixe
│   └── profil_corpus.py            # profil d'un candidat avant entraînement
├── modele/                         # entraîner
│   ├── baseline.py                 # baseline déterministe, règles et table de mots
│   ├── train_byt5.py               # affinage de ByT5
│   └── inference.py                # décodage et confiance, communs aux mesures et au service
├── evaluation/                     # mesurer
│   ├── evaluate.py                 # CER et exactitude, commande unique
│   ├── erreurs.py                  # ventilation des erreurs par catégorie
│   ├── mesures.py                  # décodage, confiance, quantification, latence
│   ├── oracle.py                   # diagnostic du classement des hypothèses
│   ├── reordonner.py               # réordonnancement des hypothèses
│   └── test_robustesse.py          # contrôle de recette ENF-08
├── service/                        # servir
│   ├── api.py                      # API REST, OpenAPI et Swagger
│   ├── normaliseur.py              # le modèle, chargé une fois
│   ├── calibration.py              # rang calibré de la confiance (PO-05)
│   ├── calibrer.py                 # calcule service/calibration.json
│   └── config.py                   # modèle, révision, limites
├── tests/                          # tests du service, sans le modèle
├── docs/                           # documentation (Sphinx, Read the Docs)
├── MODEL_CARD.md                   # carte du modèle publié sur Hugging Face
├── requirements.txt                # dépendances des données et de l'évaluation
├── requirements-train.txt          # dépendances de l'entraînement
├── requirements-service.txt        # dépendances du service
├── .readthedocs.yaml
└── .gitignore
```

**Tous les scripts se lancent depuis la racine du dépôt, en module :**

```bash
python -m evaluation.erreurs apres.csv avant.csv
```

Et non `python evaluation/erreurs.py`, qui échouerait : les scripts s'importent
les uns les autres d'un dossier à l'autre, et seul le lancement en module, depuis
la racine, rend ces imports résolubles. Les fichiers de données sont lus et
écrits dans le répertoire courant, donc à la racine également.

## Fusion et nettoyage

`merge_all.py` empile les cinq fichiers et produit `darija_all.csv`. Il retire les
lignes vides, les consignes de génération restées dans la cible et les doublons de
paires, puis ajoute des colonnes de contrôle. Le texte lui-même n'est pas réécrit :
seuls les espaces sont normalisés, car c'est ce texte qui servira à l'entraînement.

| Colonne ajoutée | Contenu |
|---|---|
| `key_ar`, `key_la` | Clés de comparaison : ponctuation, vocalisation et variantes de lettres neutralisées |
| `conflit` | Un même arabizi reçoit des cibles arabes divergentes selon la source |
| `suspect` | Les deux côtés de la paire ne se correspondent manifestement pas |

Les lignes marquées sont **conservées** et non supprimées. Deux cas ne se traitent
pas de la même façon : une même phrase arabe écrite de plusieurs façons en arabizi
est la variation que le modèle doit apprendre, alors qu'un même arabizi associé à
des cibles divergentes est une incohérence entre conventions de sources. L'arbitrage
du second cas dépend de la convention orthographique encore à fixer.

## Jeu de test et découpage

`make_splits.py` produit `train.csv` (186 861 lignes), `dev.csv` (9 799) et
`test_real.csv` (850), et garantit trois choses :

- **Le jeu de test est [`atlasia/TerjamaBench`](https://huggingface.co/datasets/atlasia/TerjamaBench)**, seul arabizi écrit par des humains dont nous disposons, le reste du corpus étant à plus de 90 % produit par des modèles. Il est réservé à l'évaluation.
- **Aucune phrase ne traverse deux blocs.** Le découpage porte sur la clé de phrase et non sur les lignes, et 110 lignes du corpus qui recoupaient le jeu de test ont été retirées.
- **Le `dev` ne contient aucune référence ambiguë**, puisqu'il sert à la mise au point.

Une colonne `darija_norm` s'ajoute : la cible débarrassée de la vocalisation,
avec le point d'interrogation en arabe. `darija` reste intact à côté. Ces deux
écarts représentaient 27 % des cibles d'entraînement contre 0,7 % de celles du
jeu de test, donc autant d'erreurs de caractères sans erreur de langue.

Les limites du jeu de test, sa taille et la discipline de son usage sont décrites
dans la [documentation](https://byt5-arabizi-normalizer.readthedocs.io/).

## Démarrage rapide

```bash
git clone https://github.com/SouleymaneDiallo04/ByT5-Arabizi-Normalizer
cd ByT5-Arabizi-Normalizer
python -m pip install -r requirements.txt

python -m donnees.extraction.extract_darija
python -m donnees.extraction.extract_doda
python -m donnees.extraction.extract_daa_pairs
python -m donnees.extraction.extract_doda_synthetic
python -m donnees.extraction.extract_doda_audio

python -m donnees.merge_all
python -m donnees.make_splits

python -m modele.baseline
python -m evaluation.evaluate
```

Les scripts lisent à distance uniquement les colonnes de texte des fichiers Parquet : il n'est pas nécessaire de télécharger les jeux de données complets (l'un d'eux pèse environ 475 Mo à cause de l'audio). Résultat : un fichier CSV par source, puis `darija_all.csv` pour le corpus fusionné.

> `extract_doda_audio` demande d'avoir accepté une fois les conditions du dépôt `atlasia/DODa-audio-dataset` sur sa page, puis d'être authentifié (`hf auth login`). `merge_all.py` signale les fichiers absents et poursuit avec les autres.

> Sous Windows, si le chemin utilisateur contient un espace, utiliser `python -m pip install ...` plutôt que `pip install ...`.

## Baseline et évaluation

La baseline est le plancher que le modèle doit dépasser. Elle n'apprend rien :
une table de correspondance de mots extraite du corpus d'entraînement, puis une
conversion caractère par caractère pour les mots qu'elle ignore.

| Système | CER | Exactitude phrase |
|---|---:|---:|
| Règles seules, aucune donnée | 0,2224 | 9,6 % |
| **Règles + table apprise sur le train** | **0,1735** | **18,2 %** |

C'est contre **0,1735** que ByT5 doit être comparé, les deux utilisant les mêmes
données. L'écart mesurera ce que le modèle apporte au-delà d'une recherche dans
un dictionnaire.

Les métriques sont produites par une **commande unique**, `evaluate.py`, qui
servira aussi au modèle et aux modèles publics de comparaison. Le détail des
résultats, par thème et par variété régionale, est dans la
[documentation](https://byt5-arabizi-normalizer.readthedocs.io/).

## Service

Le modèle est servi par une API REST documentée au format OpenAPI (EF-04).
Il est **quantifié en entier 8 bits**, 116 couches linéaires sur 129 : la tête
de sortie et la sortie des feed-forward de l'encodeur restent en pleine
précision, parce qu'elles portent toute la dégradation. À travers l'API, il
répond 42 % plus vite en médiane, pour un CER de 0,1047 contre 0,1033, et
occupe 19 % de mémoire en moins en régime établi ; son pic au démarrage
atteint toutefois 3 Go (journal du 26 novembre). `QUANTIFIE=0` sert la pleine
précision.

| Route | Rôle |
|---|---|
| `POST /normaliser` | une phrase : `{"texte": "chrit 3 tomobilat"}` |
| `POST /normaliser/lot` | de 1 à 64 phrases, résultats dans le même ordre (EF-06) |
| `GET /sante` | modèle, révision, variante (`int8` ou `fp32`) et appareil servis |
| `GET /docs` | page Swagger, pour appeler l'API depuis le navigateur |

### Avec Docker (ENF-01)

Seul Docker est nécessaire. Le modèle est téléchargé à la construction de
l'image, avec un jeton Hugging Face ayant accès à `sawti-nlp` ; le conteneur
démarre ensuite sans réseau ni jeton.

```bash
export HF_TOKEN=hf_...                     # PowerShell : $env:HF_TOKEN = "hf_..."
docker build --secret id=hf_token,env=HF_TOKEN -t byt5-arabizi .
docker run --rm -p 8000:8000 byt5-arabizi
```

Le jeton passe par un secret de construction : il ne reste dans aucune couche
de l'image. Une fois `docker ps` affichant `healthy`, la page Swagger est sur
<http://localhost:8000/docs>.

### Sans Docker

```bash
python -m pip install -r requirements-service.txt
uvicorn service.api:app --port 8000
```

Chaque résultat porte :

| Champ | Contenu |
|---|---|
| `id` | l'identifiant de la prédiction dans le registre de validation, ou `null` si le service n'enregistre pas (voir [Validation humaine](#validation-humaine)) |
| `sortie` | la transcription en écriture arabe |
| `confiance` | le **rang** de la confiance parmi les phrases du jeu de développement, en centièmes (PO-05). À 0,10, neuf phrases sur dix étaient plus sûres : les rangs bas désignent les sorties à faire relire |
| `confiance_brute` | la probabilité moyenne par octet produit, entre 0,92 et 1 en pratique, d'où le rang |
| `complete` | faux si la sortie a atteint la longueur maximale avant que le modèle termine la phrase |

Quelques règles, qui évitent des résultats faux sans signalement :

- **Une phrase ne dépasse pas 191 octets**, la longueur lue par le modèle à l'entraînement. Au-delà, le service répond 422 au lieu de tronquer l'entrée en silence.
- **Le modèle est figé à une révision du dépôt Hugging Face** (`service/config.py`), et le service refuse de démarrer si `service/calibration.json` a été calculé pour une autre révision ou une autre variante : la quantification déplace les confiances. `python -m service.calibrer` le recalcule (voir plus bas).
- **Transformers 5 est requis.** Le `config.json` publié ne se charge correctement qu'avec cette version ; voir le journal des décisions.
- Le dépôt Hugging Face est privé : définir `HF_TOKEN` avant de lancer le service.

La calibration se refait sur processeur, avec le torch de l'image, dans un
conteneur jetable : `pandas` et `scipy` n'y sont installés que le temps de la
commande, et le jeton Hugging Face local est monté en lecture seule pour lire
le jeu de développement, privé. Compter trois quarts d'heure pour 2 000 phrases.

```bash
docker compose build api
docker compose run --rm --no-deps -u root -e HF_HUB_OFFLINE=0 \
    -e HF_TOKEN_PATH=/run/jeton -v ~/.cache/huggingface/token:/run/jeton:ro \
    -v "$PWD:/app" --entrypoint sh api \
    -c "pip install -q pandas==2.3.3 scipy==1.18.1 && python -m service.calibrer"
docker compose build api                  # l'image embarque la nouvelle calibration
```

Les tests vérifient le contrat de l'API sans charger le modèle :

```bash
python -m pip install pytest httpx
python -m pytest
```

## Validation humaine

Toute prédiction passe par une validation humaine, qui alimente un registre
d'erreurs réutilisable pour le réentraînement. La conception est décrite dans
`docs/project/validation-humaine.md`. Trois conteneurs : le service, une base
Postgres et l'interface de validation.

```bash
export HF_TOKEN=hf_...                     # pour construire l'image du service
docker compose up --build -d
```

| Adresse | Rôle |
|---|---|
| <http://localhost:8000/docs> | le service ; chaque réponse porte l'`id` de la prédiction enregistrée |
| <http://localhost:8001> | les prédictions à valider, les moins sûres d'abord |
| <http://localhost:8001/historique> | les décisions prises, chacune réouvrable |

Exporter les corrections pour l'entraînement (`--avec-confirmations` ajoute les
sorties confirmées). Les phrases du jeu de test sont retirées de l'export :

```bash
docker compose run --rm --no-deps -v "$PWD:/travail" -w /travail validation \
    python -m registre.exporter registre.csv --test test_real.csv
```

Sous PowerShell, écrire `"${PWD}:/travail"`. Sous Git Bash, préfixer la commande
par `MSYS_NO_PATHCONV=1`, sans quoi `/travail` est converti en chemin Windows.

Le registre survit à `docker compose down`. Il est effacé par
`docker compose down -v`.

## Documentation

La documentation complète est disponible sur Read the Docs : <https://byt5-arabizi-normalizer.readthedocs.io/>

Pour la construire en local :

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

## Limites connues

- Les conventions d'arabizi diffèrent selon les sources (la colonne `source` permet de les séparer).
- Les données synthétiques n'ont pas été relues exhaustivement par des locuteurs natifs.
- Pas encore de jeu de test réel, ni de phrases mêlant darija et français.

## Feuille de route

- [x] Extraction et unification de cinq sources
- [x] Carte du dataset et documentation initiale
- [x] Fusion et nettoyage du corpus (`merge_all.py`)
- [ ] Lexique de mots français fréquents et phrases mixtes darija / français
- [ ] Nouveau Testament en darija (génération puis validation)
- [x] Jeu de test et découpage sans fuite (`make_splits.py`)
- [x] Baseline déterministe et harnais d'évaluation
- [ ] Jeu de contrôle d'arabizi réel de messagerie, annoté à la main
- [ ] Entraînement et évaluation du modèle ByT5

## Contributeurs

- **Hinimdou Morsia Guitdam**
- **Souleymane Diallo**

## Licence

Le corpus fusionné suit la licence la plus restrictive de ses sources, **CC BY-NC 4.0** (usage non commercial). Consulter la licence de chaque jeu de données d'origine avant toute réutilisation. La licence du code reste à définir.

## Remerciements

Ce travail repose sur les jeux de données publiés par `DrIAmed`, `MBZUAI-Paris`, `OiQ` et `Haitam03` sur Hugging Face.
