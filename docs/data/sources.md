# Sources de données

| Fichier produit | Source d'origine | Lignes | Colonnes d'origine → colonnes finales |
|---|---|---|---|
| `darija_translit.csv` | [`DrIAmed/darija-youtube-dataset`](https://huggingface.co/datasets/DrIAmed/darija-youtube-dataset) | 2 706 | `darija_Arab_new` → `darija`, `darija_Latn` → `transliteration` |
| `doda_translit.csv` | [`MBZUAI-Paris/DODa-10K`](https://huggingface.co/datasets/MBZUAI-Paris/DODa-10K) | 10 000 | `Darija_arabic` → `darija`, `Darija_latin` → `transliteration` |
| `daa_pairs_translit.csv` | [`OiQ/daa-pairs`](https://huggingface.co/datasets/OiQ/daa-pairs) | 112 814 | `arabic` → `darija`, `arabizi` → `transliteration` |
| `doda_synthetic_translit.csv` | [`Haitam03/doda_synthetic`](https://huggingface.co/datasets/Haitam03/doda_synthetic) | 64 260 | `corrected_output` → `darija`, `latin_input` → `transliteration` |
| `doda_audio_translit.csv` | [`atlasia/DODa-audio-dataset`](https://huggingface.co/datasets/atlasia/DODa-audio-dataset) | 12 743 | `darija_Arab_new` → `darija`, `darija_Latn` → `transliteration` |

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

## DODa-audio-dataset-V3

Transcriptions d'enregistrements audio. Deux colonnes de cible arabe sont
proposées, `darija_Arab_new` et `darija_Arab_old` ; seule la première est
extraite, et la colonne `audio` n'est pas lue. L'accès au dépôt est soumis à une
demande, accordée automatiquement.

Appartenant à la famille DODa, cette source recoupe les deux autres : 1 055
phrases communes avec `DODa-10K` et 1 832 avec `doda_synthetic`, mesurées après
neutralisation de la ponctuation et de la vocalisation.

```{important}
`Extractor/extract_doda_audio.py` remplace cette source par
[`atlasia/DODa-audio-dataset`](https://huggingface.co/datasets/atlasia/DODa-audio-dataset),
qui la contient entièrement : les 8 697 paires conservées de la V3 s'y
retrouvent toutes, et **3 668 s'y ajoutent**. Le pipeline redevient
intégralement reproductible, ce qu'exige ENF-04.

Les deux colonnes arabes constituent par ailleurs une ressource en elles-mêmes.
`darija_Arab_old` est une transcription phonétique, `darija_Arab_new` une
orthographe conventionnelle, et elles divergent sur **12 052 phrases sur
12 721**. Leur alignement rend 1 999 substitutions attestées — `لييا` → `ليا`,
`هووا` → `هو`, `والاكين` → `ولكن` — soit une table de normalisation produite par
des annotateurs et non par une heuristique.

Appliquée des deux côtés au moment de la mesure, elle vaut +3,5 points
d'exactitude au modèle et +4,6 à la baseline, pour un CER légèrement dégradé.
Elle **n'est pas retenue** comme métrique principale pour cette raison même ;
l'arbitrage est détaillé en PO-01 du cahier des charges.
```

```{warning}
Aucune licence n'est affichée sur la page de `doda_synthetic`, et nous n'avons
pas vérifié celle de `darija-youtube-dataset`. À vérifier avant toute
redistribution.
```
