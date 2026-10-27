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

## Fichier fusionné (à venir)

`darija_all.csv` empilera les quatre fichiers, retirera les lignes dont
`darija` ou `transliteration` est vide et supprimera les doublons exacts, sans
modifier le texte. Ce fichier n'existe pas encore : la fusion et le nettoyage
constituent l'étape suivante du projet.
