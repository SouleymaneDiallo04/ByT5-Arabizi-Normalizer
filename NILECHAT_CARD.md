---
license: other
language:
- ary
tags:
- darija
- arabizi
- transliteration
pretty_name: Paires arabizi-darija tirées de NileChat (privé)
---

# Paires arabizi – darija tirées de NileChat

> **Dépôt privé, réservé à l'équipe du projet ByT5-Arabizi-Normalizer. Il ne
> doit jamais devenir public, et son contenu ne doit être republié nulle part.**

## Pourquoi privé

Ces paires sont dérivées de deux corpus de NileChat :

| côté | source | licence |
|---|---|---|
| arabizi | [`UBC-NLP/nilechat-arabizi-mor`](https://huggingface.co/datasets/UBC-NLP/nilechat-arabizi-mor) | CC-BY-NC-4.0 |
| arabe | [`UBC-NLP/nilechat-lhv-mor`](https://huggingface.co/datasets/UBC-NLP/nilechat-lhv-mor) | **CC-BY-ND-4.0** |

La licence CC-BY-ND interdit de distribuer une œuvre dérivée, et un corpus de
paires en est une. Le dépôt sert à partager le fichier au sein de l'équipe sans
que chacun ait à le reconstruire ; il ne vaut pas autorisation de le diffuser.

## Contenu

`nilechat_translit.csv`, le fichier produit par l'extraction, tel quel. Mêmes
colonnes que les autres sources du projet :

| colonne | contenu |
|---|---|
| `id` | identifiant du document NileChat d'origine |
| `darija` | la phrase en écriture arabe |
| `transliteration` | la même phrase en arabizi |
| `source` | `nilechat` |

## Ce que ce n'est pas

**L'arabizi est produit par un modèle de langue**, pas écrit par des humains. Ne
jamais présenter ces paires comme de l'arabizi réel.

**Le fichier n'est pas purgé du jeu de test.** `donnees/melanger.py` retire les
phrases qui recoupent TerjamaBench avant tout entraînement (628 lors du
mélange 3). Ne pas entraîner sur ce fichier sans passer par lui.

## Construction

`python -m donnees.extraction.extract_nilechat`, dans le dépôt GitHub du projet.
Le script joint l'arabizi à sa source arabe sur les identifiants de documents,
découpe les documents en phrases en ne gardant que les lignes alignées des deux
côtés, écarte celles dont le rapport de longueur est aberrant ou qui dépassent
les longueurs du modèle, et ne garde que les cibles portant un marqueur propre
au darija (`ديال`, `اللي`, `دابا`…), pour exclure l'arabe standard.

Il se relance à l'identique par quiconque a accepté les conditions des deux
dépôts NileChat.

## Utilisation

```bash
hf download sawti-nlp/nilechat-arabizi-pairs nilechat_translit.csv --repo-type dataset --local-dir .
python -m donnees.melanger nilechat_translit.csv
```

Le mélange avec le corpus du projet, ses proportions et leurs effets sont
décrits dans le journal des décisions de la documentation.
