# Sources de données

| Fichier produit | Source d'origine | Lignes | Colonnes d'origine → colonnes finales |
|---|---|---|---|
| `darija_translit.csv` | [`DrIAmed/darija-youtube-dataset`](https://huggingface.co/datasets/DrIAmed/darija-youtube-dataset) | 2 706 | `darija_Arab_new` → `darija`, `darija_Latn` → `transliteration` |
| `doda_translit.csv` | [`MBZUAI-Paris/DODa-10K`](https://huggingface.co/datasets/MBZUAI-Paris/DODa-10K) | 10 000 | `Darija_arabic` → `darija`, `Darija_latin` → `transliteration` |
| `daa_pairs_translit.csv` | [`OiQ/daa-pairs`](https://huggingface.co/datasets/OiQ/daa-pairs) | 112 814 | `arabic` → `darija`, `arabizi` → `transliteration` |
| `doda_synthetic_translit.csv` | [`Haitam03/doda_synthetic`](https://huggingface.co/datasets/Haitam03/doda_synthetic) | 64 260 | `corrected_output` → `darija`, `latin_input` → `transliteration` |

## darija-youtube-dataset

Segments issus de transcriptions de vidéos YouTube. La translittération suit
une convention proche de Buckwalter (`T` pour ط, `sh`, `dh`), différente de
l'arabizi de messagerie. Les segments consécutifs se chevauchent souvent.

## DODa-10K

Arabizi marocain avec les chiffres usuels (`3` pour ع, `7` pour ح, `9` pour ق).
Licence affichée sur la page d'origine : CC BY-NC 4.0.

## daa-pairs

Données synthétiques générées par un LLM à partir de transcriptions YouTube et
d'articles. L'arabizi est peu homogène (tirets et apostrophes variables), et
quelques lignes mélangent des caractères arabes dans la colonne latine. Le
dataset contient aussi une colonne `mixte` (darija en écriture arabe avec les
mots étrangers laissés en latin), non extraite pour l'instant mais prévue pour
amorcer le lexique français. Licence affichée : Apache 2.0.

## doda_synthetic

Données synthétiques. `latin_input` est de l'arabizi brut, avec fautes et
variantes, ce qui correspond à l'entrée d'un normaliseur ; `corrected_output`
est la version arabe corrigée par un modèle. Le dataset contient aussi
`normalized_latin_darija` (arabizi normalisé, avec une convention différente :
`8` pour ه, `5` pour خ), non utilisée pour l'instant. Le nom suggère un lien avec
DODa : des doublons avec DODa-10K sont possibles.

```{warning}
Aucune licence n'est affichée sur la page de `doda_synthetic`, et nous n'avons
pas vérifié celle de `darija-youtube-dataset`. À vérifier avant toute
redistribution.
```
