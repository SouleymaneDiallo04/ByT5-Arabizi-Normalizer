# Modèle

Le normaliseur est **`google/byt5-small`**, 300 millions de paramètres, affiné
pour transcrire l'arabizi en darija en écriture arabe. ByT5 lit et écrit des
octets : il n'a pas de vocabulaire figé, ce qui convient à une écriture aussi
variable que l'arabizi.

## Modèles produits

| modèle | méthode | CER | exactitude | dossier Hugging Face |
|---|---|---:|---:|---|
| baseline | règles et table, {doc}`baseline` | 0,1699 | 17,6 % | — |
| premier entraînement | affinage complet, corpus du projet | 0,1087 | 29,4 % | `modèles_S/` |
| **mélange 3** | affinage complet, corpus du projet et nilechat | **0,1033** | **32,2 %** | racine, **recommandé** |

Le service sert le mélange 3 quantifié en entier 8 bits : CER 0,1047 et 31,3 %
d'exactitude à travers l'API, pour une latence réduite de 42 % (journal du
26 novembre).

Mesures sur les 678 phrases de TerjamaBench réservées à l'évaluation ; les
métriques et la discipline d'usage du jeu de test sont décrites dans
{doc}`evaluation`. Le cheminement d'un modèle à l'autre, expériences sans effet
comprises, est retracé dans le {doc}`journal des décisions <../project/decisions>`.

Deux modèles affinés par adaptateurs LoRA sont documentés dans {doc}`lora`,
dossiers `ModelOneAdaptateur/` et `ModèleMultipleAdaptateur /`. Ils sont mesurés
sur un autre jeu, `test_clean`, et ne figurent pas dans ce tableau pour cette
raison.

## Utilisation

Le dépôt Hugging Face est privé : il faut un jeton ayant accès à `sawti-nlp`.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

depot = "sawti-nlp/byt5-arabizi-normalizer"
tokenizer = AutoTokenizer.from_pretrained(depot)
modele = AutoModelForSeq2SeqLM.from_pretrained(depot)

entree = tokenizer("kanbghik bzaf", return_tensors="pt")
sortie = modele.generate(**entree, max_length=288)
print(tokenizer.decode(sortie[0], skip_special_tokens=True))
```

Décodage **glouton** : la recherche en faisceau a été mesurée et dégrade le CER.
Les limites connues du modèle, dont l'écart régional, sont détaillées dans sa
fiche, `MODEL_CARD.md` à la racine du dépôt GitHub.
