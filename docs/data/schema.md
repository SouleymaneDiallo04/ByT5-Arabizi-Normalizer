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
