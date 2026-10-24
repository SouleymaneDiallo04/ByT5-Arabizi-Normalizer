# ByT5-Arabizi-Normalizer

**ByT5-Arabizi-Normalizer** est un projet de normalisation de l'**arabizi** :
convertir du darija marocain écrit en lettres latines et en chiffres
(`3tini`, `9alb`, `7ta`) vers l'écriture arabe, malgré la grande variabilité
d'orthographe des messages réels. Le modèle repose sur **ByT5**, qui travaille
au niveau des octets et gère bien les orthographes non standard.

Cette documentation regroupe tout ce qui concerne le projet : sa présentation,
les données, le modèle et la manière de contribuer. Elle évolue avec le projet.

```{note}
Version préliminaire. Les sections marquées « À compléter » seront remplies au
fur et à mesure de l'avancement.
```

## Où commencer ?

```{list-table}
:header-rows: 1

* - Je veux...
  - Aller à
* - Comprendre le projet
  - {doc}`project/overview`
* - Reproduire le jeu de données
  - {doc}`getting-started/quickstart`
* - Connaître les sources de données
  - {doc}`data/sources`
* - Voir ce qui est prévu
  - {doc}`project/roadmap`
* - Contribuer
  - {doc}`contributing`
```

```{toctree}
:maxdepth: 2
:caption: Démarrage

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Projet

project/overview
project/roadmap
project/contributors
```

```{toctree}
:maxdepth: 2
:caption: Données

data/index
data/sources
data/schema
data/reproduction
data/limitations
```

```{toctree}
:maxdepth: 2
:caption: Modèle

model/index
```

```{toctree}
:maxdepth: 1
:caption: Ressources

contributing
changelog
glossary
```
