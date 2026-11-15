---
language:
- ary
license: cc-by-nc-4.0
base_model: google/byt5-small
library_name: transformers
tags:
- darija
- moroccan-arabic
- arabizi
- transliteration
- byt5
datasets:
- sawti-nlp/darija-arabizi-corpus
metrics:
- cer
---

# ByT5-Arabizi-Normalizer

Transcrit l'**arabizi marocain** — le darija écrit en lettres latines, avec des
chiffres pour les sons absents du français — en **darija en écriture arabe**.

```
chrit 3 tomobilat        ->  شريت 3 طوموبيلات
fin ghadi daba           ->  فين غادي دابا؟
kanbghik bzaf            ->  كنبغيك بزاف
```

Le modèle change l'alphabet. Il ne traduit pas, et ne doit pas ramener une forme
régionale vers celle de Casablanca.

## Quel modèle utiliser

Ce dépôt contient plusieurs modèles. **Celui de la racine est le modèle
recommandé.**

| emplacement | modèle | statut |
|---|---|---|
| **racine** | ByT5-small affiné, mélange 3 | **recommandé** |
| `modèles_S/` | ByT5-small affiné, premier entraînement | conservé pour rejouer les résultats antérieurs |
| `ModelOneAdaptateur/` | ByT5-small, adaptateur LoRA unique | documenté par son auteur |
| `ModèleMultipleAdaptateur /` | ByT5-small, cinq adaptateurs LoRA | documenté par son auteur |

Les deux modèles LoRA ont été évalués sur un jeu de test différent du nôtre ;
leurs chiffres ne sont pas comparables à ceux de cette page. Leur méthode et
leurs résultats sont décrits dans la documentation du projet et dans les
carnets du dépôt de données.

Attention au nom du dernier dossier, qui se termine par une espace.

## Utilisation

Le dépôt est privé : il faut un jeton Hugging Face ayant accès à `sawti-nlp`.

**Transformers 5 est requis**, le modèle a été vérifié avec la version 5.20.2.
Au chargement, un avertissement signale que la tête de sortie n'est pas liée aux
plongements : c'est le comportement voulu, ByT5 les garde distinctes. Ne pas
corriger `tie_word_embeddings` dans `config.json` : à `false`, les plongements
de l'encodeur et du décodeur ne sont plus chargés.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

depot = "sawti-nlp/byt5-arabizi-normalizer"
tokenizer = AutoTokenizer.from_pretrained(depot)
modele = AutoModelForSeq2SeqLM.from_pretrained(depot)

entree = tokenizer("chrit 3 tomobilat", return_tensors="pt")
sortie = modele.generate(**entree, max_length=288)
print(tokenizer.decode(sortie[0], skip_special_tokens=True))
```

**Décodage glouton**, comme ci-dessus. La recherche en faisceau a été mesurée
et dégrade le CER : elle maximise la vraisemblance de la phrase, qui n'est pas
la métrique. Quatre méthodes de réordonnancement des hypothèses ont été
essayées, aucune ne bat le décodage glouton.

Pour charger le premier modèle, à fin de comparaison :

```python
modele = AutoModelForSeq2SeqLM.from_pretrained(depot, subfolder="modèles_S")
```

## Résultats

Mesurés sur 678 phrases de TerjamaBench réservées à l'évaluation, qui n'ont
servi à aucun réglage. 172 autres en ont été prélevées pour régler les
décisions de sélection.

| modèle | CER | exactitude phrase |
|---|---:|---:|
| baseline déterministe, règles et table | 0,1699 | 17,6 % |
| premier entraînement | 0,1087 | 29,4 % |
| **ce modèle** | **0,1033** | **32,2 %** |

Le CER est calculé au niveau du corpus, somme des distances d'édition sur somme
des longueurs de référence. L'exactitude est la part de phrases rendues
caractère pour caractère. Les deux sont mesurés après normalisation de la
vocalisation et de la ponctuation interrogative, des deux côtés.

Un mot français laissé en latin au lieu d'être translittéré : **une phrase sur
678**, contre quatre pour le premier entraînement. La référence du jeu de test
en porte une aussi.

## Entraînement

`google/byt5-small`, affiné en précision fp16 sur deux T4, **15 000 pas de 64
exemples**, décodage et sélection décrits plus haut. La perte de validation
descendait encore au dernier contrôle, 0,1155 : le modèle n'a pas convergé.

Les données sont un mélange, à proportions fixées à l'avance :

| part | source | contenu |
|---:|---|---|
| 58,4 % | corpus du projet | 186 861 paires de cinq sources publiques, nettoyées, chacune vue trois fois |
| 41,6 % | nilechat | 399 360 paires tirées de 3 148 159 |

**L'arabizi de nilechat est produit par un modèle de langue**, pas écrit par des
humains. Il a été joint à sa source arabe sur les identifiants des documents,
découpé en phrases sous contrôle d'alignement, et filtré pour ne garder que des
cibles en darija et non en arabe standard. 628 phrases recoupant le jeu de test
en ont été retirées avant l'entraînement.

Les deux parts ne jouent pas le même rôle, et ça a été mesuré. La part de
nilechat améliore la cohérence des voyelles longues : ses cibles, produites par
un même modèle, sont écrites de façon homogène. L'exposition au corpus du
projet commande la translittération des mots français : la réduire de moitié
fait réapparaître le latin dans les sorties.

## Limites connues

**L'écart régional.** Le modèle reste nettement moins bon sur les variétés du
Sud et du Rif que sur celles du centre.

| variété de l'annotateur | phrases | CER |
|---|---:|---:|
| Marrakech | 293 | 0,0945 |
| Rabat | 27 | 0,0750 |
| Nador | 17 | 0,1507 |
| Guelmim | 20 | 0,1806 |

Ce modèle a réduit l'écart — Guelmim était à 0,2159 au premier entraînement —
sans le combler. Les échantillons de ces deux variétés sont petits, et ces
chiffres sont indicatifs. La cause mesurée est lexicale : une part importante
des formes attendues n'apparaît jamais dans les données d'entraînement.

**La régularisation dialectale.** Face à une forme régionale, le modèle tend à
produire la forme majoritaire : `bit` devient `بغيت` au lieu de `بيت`. C'est le
défaut principal relevé à l'analyse d'erreurs.

**Les mots étrangers récents** n'ont pas d'orthographe arabe établie. `gaming`
s'écrit aussi bien `الغيمينج` que `الڭامينغ`, et une partie de l'erreur sur ce
thème n'est pas réductible.

**La convention de sortie** est celle de TerjamaBench : sans vocalisation, point
d'interrogation arabe. Un utilisateur attendant une autre convention verra des
écarts qui ne sont pas des erreurs de langue.

**Le score de confiance** n'est pas produit par le modèle lui-même. La
probabilité moyenne par octet se concentre entre 0,92 et 1 et ne se lit pas
directement ; le service doit renvoyer son rang, calibré sur des données de
réglage.

## Licence

**CC-BY-NC-4.0 — usage non commercial.**

Le modèle est entraîné en partie sur des données dérivées de nilechat. Son
arabizi est publié sous CC-BY-NC-4.0, et sa source arabe sous **CC-BY-ND-4.0**,
qui interdit de distribuer une œuvre dérivée. Savoir si un modèle entraîné sur
ces données en est une n'est pas tranché. Le dépôt est privé ; **rendre ce
modèle public demande d'avoir réglé cette question au préalable.**

Le corpus du projet est publié sous CC-BY-NC-4.0.

## Reproduire

Tout le code est public :
<https://github.com/SouleymaneDiallo04/ByT5-Arabizi-Normalizer>

Le journal des décisions de la documentation retrace chaque expérience, y
compris celles qui n'ont rien donné, et la raison de chacun des choix ci-dessus.
