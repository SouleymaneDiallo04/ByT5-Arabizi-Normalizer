# Présentation du projet

## Problème

Sur les réseaux sociaux et les messageries, le darija marocain est très souvent
écrit en **arabizi** : lettres latines complétées par des chiffres qui
représentent des sons arabes (`3` pour ع, `7` pour ح, `9` pour ق). Il n'existe
pas d'orthographe standard : un même mot peut s'écrire de plusieurs façons
(`merci`, `mersi`, `mrc`), ce qui complique le traitement automatique.

## Objectif

Entraîner un modèle qui **normalise** l'arabizi, c'est-à-dire qui le convertit
en darija écrit en caractères arabes, de façon robuste face aux fautes, aux
variantes d'écriture et aux abréviations.

## Approche

- **Modèle** : ByT5, un modèle de type encodeur-décodeur qui travaille sur les
  octets plutôt que sur des mots ou des sous-mots. Il n'a donc pas de
  vocabulaire fermé et supporte bien les orthographes non standard.
- **Données** : un corpus de paires darija en écriture arabe ↔ arabizi,
  rassemblé à partir de quatre jeux de données publics (voir
  {doc}`../data/sources`).
- **Cas particulier** : les messages qui mélangent darija et français.
  Un lexique de mots français fréquents et des phrases mixtes dédiées sont
  prévus pour que le modèle ne soit pas pris au dépourvu (voir
  {doc}`roadmap`).

## État actuel

Les données sont extraites et unifiées. L'entraînement du modèle n'a pas encore
commencé. Voir {doc}`roadmap`.
