# Installation

## Prérequis

- Python 3.10 ou plus récent
- Git

## Récupérer le dépôt

```bash
git clone https://github.com/SouleymaneDiallo04/ByT5-Arabizi-Normalizer
cd ByT5-Arabizi-Normalizer
```

## Dépendances pour extraire les données

```bash
python -m pip install pandas pyarrow requests huggingface_hub
```

```{tip}
Sous Windows, si ton chemin utilisateur contient un espace, la commande
`pip install ...` peut échouer avec « Unable to create process ». Utilise
toujours `python -m pip install ...`.
```

## Dépendances pour construire cette documentation

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Ouvre ensuite `docs/_build/html/index.html` dans un navigateur.

```{note}
Les dépendances d'entraînement du modèle seront ajoutées ici. À compléter.
```
