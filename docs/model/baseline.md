# Baseline déterministe

La baseline répond à l'exigence **EF-05**. Elle sert de plancher : le modèle
ByT5 n'a d'intérêt que s'il la dépasse nettement. Elle n'apprend rien et
n'ajuste aucun paramètre.

```bash
python baseline.py       # produit baseline_predictions.csv
python evaluate.py       # calcule les métriques
```

## Comment elle fonctionne

Deux étages, dans cet ordre.

**Une table de correspondance de mots**, apprise sur `train.csv` uniquement.
L'alignement est positionnel, donc limité aux paires dont les deux côtés ont le
même nombre de mots, soit environ 37 % du corpus : l'arabe agglutine ses
préfixes (`f lmdina` donne `فالمدينة`), ce qui écarte les autres. 11 260 mots
sont retenus, couvrant 65 % des mots du jeu de test.

**Une conversion caractère par caractère** pour les 35 % de mots restants :
digrammes d'abord (`ch` → `ش`), puis chiffres (`3` → `ع`), puis lettres. Un
chiffre formant un mot à lui seul reste un nombre, conformément à **EF-02**.

L'article défini est retiré de la forme arabe quand l'arabizi ne le porte pas.
Sans cela la table apprend la forme la plus fréquente du corpus et colle un
article de trop, ce qui faisait échouer le cas de test d'EF-02.

## Résultats sur le jeu de test

| Système | CER | Exactitude phrase |
|---|---:|---:|
| Règles seules, aucune donnée | 0,2224 | 9,6 % |
| **Règles + table apprise sur le train** | **0,1735** | **18,2 %** |

La table seule apporte 0,049 de CER et 8,6 points d'exactitude. C'est donc
contre **0,1735** que ByT5 doit être comparé, puisque les deux utilisent les
mêmes données d'entraînement. L'écart entre les deux mesurera ce que le modèle
apporte au-delà d'une recherche dans un dictionnaire.

| Par type | Phrases | CER | Exactitude |
|---|---:|---:|---:|
| Mots isolés | 163 | 0,2055 | 48,5 % |
| Phrases | 687 | 0,1723 | 11,1 % |

| Par thème | Phrases | CER | Exactitude |
|---|---:|---:|---:|
| common_phrases | 136 | 0,1195 | 23,5 % |
| numeric_and_date | 62 | 0,1358 | 16,1 % |
| long_sentences | 50 | 0,1418 | 0 % |
| idioms | 51 | 0,1580 | 13,7 % |
| named_entities | 53 | 0,1727 | 0 % |
| single_words | 163 | 0,2055 | 48,5 % |
| educational | 73 | 0,2069 | 1,4 % |
| dialect_variation | 46 | 0,2077 | 19,6 % |
| humor | 50 | 0,2098 | 12,0 % |
| religion | 66 | 0,2178 | 10,6 % |
| incorrect_spellings | 50 | 0,2213 | 8,0 % |
| **mixed_language** | 50 | **0,2750** | **0 %** |

`mixed_language` est la pire catégorie, et ce n'est pas un hasard : la table a
hérité de la convention de `daa-pairs`, qui laisse les mots français en lettres
latines, alors que le jeu de test les translittère phonétiquement. Vingt
prédictions contiennent encore des lettres latines pour cette raison. C'est le
point ouvert sur l'alternance codique, rendu visible dans un score.

## Ce qu'elle ne sait pas faire

Les règles échouent de trois façons systématiques, par construction :

- **Elles écrivent toutes les voyelles.** `3andek` donne `عاندك` au lieu de
  `عندك`, car `a` donne toujours `ا`.
- **Elles ignorent les emphatiques.** `s` donne toujours `س`, jamais `ص`.
- **Elles ne collent pas l'article.** `lkhir` donne `لخير` au lieu de `الخير`.

La table corrige ces trois points pour les mots qu'elle connaît, mais travaille
au niveau du mot et ne peut donc rien décider selon le contexte. C'est
précisément ce que le modèle est censé apporter.

## Score de confiance (EF-03)

Chaque normalisation est accompagnee d'un score dans `[0, 1]`, dans la colonne
`confiance`. Il vaut la moyenne des confiances par mot : pour un mot trouve dans
la table, la part de sa traduction majoritaire dans le corpus d'entrainement ;
pour un mot converti par regles, une valeur fixe de **0,25**.

Ces valeurs ne sont pas choisies, elles sont mesurees sur `dev.csv` par
alignement positionnel :

| Origine du mot | Mots | Exactitude reelle |
|---|---:|---:|
| Table apprise | 12 094 | **87,2 %** |
| Regles | 4 089 | **24,7 %** |

Et la fiabilite annoncee par la table predit bien la justesse reelle, ce qui
justifie de l'employer directement comme probabilite :

| Fiabilite annoncee | Mots | Exactitude reelle |
|---|---:|---:|
| 0,5 et plus | 554 | 53,4 % |
| 0,6 et plus | 694 | 70,2 % |
| 0,7 et plus | 851 | 73,0 % |
| 0,8 et plus | 1 885 | 85,5 % |
| 0,9 et plus | 8 110 | 92,8 % |

### Le score transfere au jeu de test

La calibration est faite sur le `dev`, mais elle se verifie sur le jeu de test,
qu'elle n'a jamais vu. Le CER y decroit de facon monotone avec la confiance :

| Confiance | Phrases | CER moyen |
|---|---:|---:|
| 0 a 40 % | 155 | 0,2925 |
| 40 a 60 % | 208 | 0,2154 |
| 60 a 80 % | 349 | 0,1560 |
| 80 a 95 % | 84 | 0,1007 |
| 95 a 100 % | 54 | **0,0471** |

Correlation de Spearman entre confiance et CER : **-0,444**.

C'est ce qui rend le score exploitable pour EF-07 : un seuil permet de router
vers un annotateur humain les seules normalisations douteuses, au lieu de tout
faire relire. Le tableau ci-dessus donne de quoi choisir ce seuil en connaissance
du taux d'erreur qu'il laisse passer.

## Critères de recette satisfaits

| Exigence | Critère | État |
|---|---|---|
| EF-01 | sortie non vide, UTF-8 valide | 850/850 |
| EF-02 | `chrit 3 tomobilat` donne `شريت 3 طوموبيلات` | passe |
| EF-05 | un résultat par entrée, métriques reproductibles | 850/850 |
| ENF-04 | deux exécutions, mêmes métriques ; versions épinglées | identiques, `requirements.txt` |
| EF-03 | score dans [0, 1] pour chaque normalisation | 850/850, de 0,250 à 1,000 |
| ENF-08 | aucune exception sur cent entrées bruitées | 109/109 |

Le contrôle d'ENF-08 est exécutable : `python test_robustesse.py`.
