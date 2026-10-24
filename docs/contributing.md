# Contribuer

## Flux de travail

1. Créer une branche à partir de `main` : `git checkout -b ma-modification`
2. Faire les modifications, puis commit avec un message clair
3. Pousser la branche et ouvrir une *pull request*
4. Faire relire par l'autre contributeur avant de fusionner

## Modifier la documentation

Les pages sont des fichiers Markdown dans le dossier `docs/`. Pour ajouter une
page, créer le fichier puis l'ajouter à un `toctree` dans `docs/index.md`.

Construire la documentation en local avant de pousser :

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Read the Docs reconstruit automatiquement la documentation à chaque push sur
la branche configurée.

## Ajouter une source de données

1. Écrire un script `extract_<source>.py` qui produit un CSV au format décrit
   dans {doc}`data/schema`
2. Ajouter la source dans {doc}`data/sources` avec ses colonnes et sa licence
3. Mettre à jour `merge_all.py` et la carte du dataset Hugging Face
4. Noter la modification dans {doc}`changelog`
