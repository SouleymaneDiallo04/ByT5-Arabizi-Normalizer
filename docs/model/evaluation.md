# Évaluation

Les métriques sont produites par **une seule commande**, comme l'exige le cahier
des charges. La baseline, le modèle ByT5 et les modèles publics auxquels nous
nous comparons passent tous par elle : si chacun calculait son propre CER, les
chiffres ne seraient pas comparables.

```bash
python evaluate.py [fichier_de_predictions.csv]
```

Le fichier attendu contient au minimum les colonnes `prediction`, `darija` et
`darija_norm`. Les colonnes `mot_isole`, `annotator_dialect` et `topic`, quand
elles sont présentes, déclenchent les ventilations correspondantes.

## Les deux métriques

**Le CER** est le taux d'erreur de caractères : distance de Levenshtein entre la
prédiction et la référence, divisée par la longueur de la référence. Il est
calculé **au niveau du corpus**, somme des distances sur somme des longueurs.
Une moyenne des CER par phrase donnerait un poids excessif aux phrases courtes,
dont le jeu de test contient beaucoup.

**L'exactitude phrase** est la part des prédictions identiques à la référence,
caractère pour caractère.

```{important}
Avec 850 items, l'intervalle de confiance à 95 % sur l'exactitude phrase est
d'environ **3,4 points**. Deux systèmes séparés par moins que cela ne sont pas
distinguables. Le CER, mesuré au caractère sur des milliers de caractères, est
nettement plus fiable : c'est lui qu'il faut traiter comme métrique principale.
```

## Deux références, et ce que leur écart raconte

Le script mesure contre deux références :

- **le texte brut**, tel qu'il figure dans le jeu de test ;
- **la convention appliquée des deux côtés**, prédiction comprise : vocalisation
  retirée, point d'interrogation en arabe.

L'écart entre les deux chiffre ce que la convention orthographique coûte, et
protège de l'accusation d'avoir mesuré du bruit. Sur la baseline il est nul
(+0,0002), ce qui est normal : une table de règles ne produit jamais de
diacritiques. Il prendra son sens avec ByT5, qui apprendra la vocalisation de
ses données d'entraînement alors que le jeu de test n'en contient pas.

## Discipline d'usage du jeu de test

```{warning}
Un jeu de test n'est verrouillé que si l'on cesse de le regarder. Avec 850
items, quelques tours de « voyons ce que le modèle a raté et ajustons » suffisent
à l'éroder sans que rien ne le signale.
```

Les mesures de mise au point se font sur `dev.csv`. Le jeu de test ne s'ouvre
qu'aux jalons décidés à l'avance, et chaque ouverture est consignée dans le
{doc}`../changelog` avec sa date et son objet.
