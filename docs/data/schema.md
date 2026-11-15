# Format des fichiers

Tous les fichiers ont le même schéma : CSV, UTF-8 avec BOM (`utf-8-sig`),
séparateur virgule, une ligne d'en-tête.

| Colonne | Type | Contenu |
|---|---|---|
| `id` | texte | Identifiant unique, préfixé selon la source |
| `darija` | texte | Phrase en darija, écriture arabe |
| `transliteration` | texte | La même phrase en lettres latines (arabizi) |
| `source` | texte | Jeu de données d'origine |

## Préfixes d'identifiants

| Source | Préfixe | Exemple |
|---|---|---|
| darija-youtube-dataset | `darija-youtube_` | `darija-youtube_00000` |
| DODa-10K | `doda10k_` | `doda10k_00000` |
| daa-pairs | `daa-pairs_` | `daa-pairs_000000` |
| doda_synthetic | `doda-synthetic_` | `doda-synthetic_00000` |

Pour DODa-10K, le numéro reprend l'`index` d'origine. Pour les autres sources,
c'est la position dans le fichier.

## Exemples

Tirés de `Haitam03/doda_synthetic` :

| darija | transliteration |
|---|---|
| سمح ليا | sma7 liya |
| أنا مزوّج | ana mzawaj |
| غانمشيو فيها كاملين | ghanmchiw fiha kamlin |

## Fichier fusionné

`darija_all.csv`, produit par `merge_all.py`, empile les cinq fichiers et
conserve les quatre colonnes ci-dessus, complétées par quatre colonnes de
contrôle.

| Colonne | Type | Contenu |
|---|---|---|
| `key_ar` | texte | Clé de comparaison de `darija` : ponctuation, vocalisation et variantes de lettres neutralisées |
| `key_la` | texte | Clé de comparaison de `transliteration` : minuscules, ponctuation neutralisée |
| `conflit` | booléen | Une même `key_la` reçoit plusieurs `key_ar` distinctes |
| `suspect` | booléen | Rapport de longueur entre les deux côtés hors de la bande plausible |

Le texte des colonnes `darija` et `transliteration` n'est pas réécrit : seuls les
espaces sont normalisés. Les clés servent uniquement à comparer, jamais à
entraîner.

### Ce que la fusion retire

| Motif | Lignes |
|---|---|
| `darija` ou `transliteration` vide | 270 |
| Consigne de génération restée dans la cible | 1 |
| Doublons de paires, aux variantes d'écriture près | 4 256 |

### Ce qu'elle marque sans le retirer

| Motif | Lignes |
|---|---|
| `conflit` | 21 711 |
| `suspect` | 188 |

Une même phrase arabe écrite de plusieurs façons en arabizi n'est **pas** un
doublon : c'est la variation d'écriture que le modèle doit apprendre à absorber.
Seule la paire entièrement identique est supprimée.

## Fichiers d'entraînement

`make_splits.py` tire trois fichiers du corpus fusionné.

| Fichier | Lignes | Contenu |
|---|---:|---|
| `train.csv` | 186 861 | Corpus d'entraînement |
| `dev.csv` | 9 799 | Mise au point et arrêt précoce |
| `test_real.csv` | 850 | TerjamaBench, évaluation seule |

Une colonne s'ajoute, et une seule :

| Colonne | Contenu |
|---|---|
| `darija_norm` | La cible, vocalisation retirée et point d'interrogation en arabe |

`darija` reste intact à côté, ce qui rend la normalisation réversible et
permet de l'évaluer en ablation. Elle ne touche que les points non débattus :
TerjamaBench ne porte pas de vocalisation (0,2 % de ses mots contre 21 % dans
`doda_synthetic`) et écrit `؟` dans 100 % des cas contre 3 % pour `DODa-10K`.
Laisser ces écarts ferait compter chaque diacritique produit comme une erreur de
caractère, sans la moindre erreur de langue. Elle modifie 27 % des cibles
d'entraînement et seulement 0,7 % de celles du jeu de test.

Le jeu de test ajoute `topic`, `annotator_dialect` et `mot_isole`, pour permettre
l'analyse d'erreurs par thème et par variété régionale.

### Trois garanties

- **Aucune phrase commune entre `train` et `dev`.** Le découpage porte sur
  `key_ar`, pas sur les lignes : une phrase présente dans deux sources reste
  d'un seul côté.
- **Aucune phrase commune entre `train` et le jeu de test.** 110 lignes du
  corpus ont été retirées à ce titre, 91 par le côté arabe, 38 par l'arabizi,
  19 par les deux.
- **Aucune ligne ambiguë dans `dev`.** Les lignes `conflit` et `suspect` en sont
  écartées, une référence de mise au point devant être unique.

```{note}
Ce dernier choix a un coût : les conflits étant concentrés sur la famille DODa,
les écarter du `dev` y réduit mécaniquement la part de `DODa-10K` (0,6 % contre
5,1 % dans `train`). Le `dev` est donc propre mais non représentatif par source.
Le compromis est assumé : le `dev` sert l'arrêt précoce, il doit donc ressembler
au jeu de test, dont `daa-pairs` est justement la source la plus proche.
```
