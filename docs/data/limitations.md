# Limites et précautions

- **Conventions d'arabizi différentes** selon les sources. La colonne `source`
  permet de les séparer ou de les pondérer avant l'entraînement.
- **Données synthétiques.** `daa-pairs` et `doda_synthetic` ont été produites ou
  corrigées par des modèles de langage, sans relecture exhaustive par des
  locuteurs natifs. Elles peuvent contenir des erreurs. Il est conseillé de les
  traiter à part de `DODa-10K` pour l'évaluation.
- **Recoupements entre sources, mesurés.** Les trois sources de la famille DODa
  se recoupent : 1 055 phrases communes entre `DODa-10K` et `DODa-audio-V3`,
  1 832 entre `doda_synthetic` et `DODa-audio-V3`, 742 entre `DODa-10K` et
  `doda_synthetic`. `merge_all.py` retire les doublons de paires, mais une même
  phrase peut rester présente avec deux graphies arabizi différentes, ce qui est
  voulu.
- **Cibles divergentes.** 21 711 lignes reçoivent, pour un même arabizi, des
  cibles arabes différentes selon la source. 99 % de ces désaccords sont
  inter-sources : chaque source est cohérente avec elle-même mais applique sa
  propre convention. Ces lignes portent le drapeau `conflit` et ne sont pas
  supprimées, leur arbitrage dépendant de la convention cible à fixer.
- **Pas de messages de messagerie réels.** Le corpus ne contient ni messages
  mixtes darija et français dans la même phrase, ni texte provenant de vraies
  conversations.
- **Licences hétérogènes.** Voir {doc}`sources`. Le corpus fusionné est publié
  sous CC BY-NC 4.0, la plus restrictive des licences connues.

## Fichier interdit comme matière première

```{danger}
`raw/atlaset/data/test-00000-of-00001.parquet`, sur le dépôt Hugging Face, ne
doit **jamais** servir à produire des données d'entraînement, ni à générer de
l'arabizi synthétique, ni à amorcer un lexique.
```

830 de ses 2 683 lignes **sont des lignes de TerjamaBench**, c'est-à-dire du jeu
de test : même phrase arabe, même arabizi sous la clé `darija_Latn` de la colonne
`metadata`, mêmes champs `topic` et `annotator_dialect`. Cela représente 97,6 %
du jeu de test. L'utiliser revient à entraîner le modèle sur sa propre
évaluation, et rien ne le signalerait : les scores seraient simplement faux.

La partition `train` d'Atlaset reste utilisable **après filtrage explicite**. Sur
un échantillon de 21 515 lignes, 2 coïncidaient avec TerjamaBench, soit de
l'ordre d'une centaine sur l'ensemble. Le filtre doit être appliqué, pas supposé.

Son registre est par ailleurs éloigné de la tâche : 23 mots de médiane contre 4
dans le jeu de test, avec de la presse, de l'encyclopédie et des jeux
d'instruction anglais traduits automatiquement. Une seule de ses sources est
conversationnelle, pour environ 4,6 % du total.

## Le jeu de test

`test_real.csv` provient de [`atlasia/TerjamaBench`](https://huggingface.co/datasets/atlasia/TerjamaBench) :
850 lignes annotées à la main, huit variétés régionales. C'est le seul arabizi
écrit par des humains dont nous disposons, le reste étant à plus de 90 % produit
par des modèles de langage.

Trois limites à connaître avant d'interpréter un score :

- **Ce n'est pas de l'arabizi de messagerie** mais d'annotateur. 4,7 % des lignes
  portent une ponctuation finale contre 62,7 % dans le corpus, 1,1 % une
  apostrophe contre 21,4 %, et aucune ne contient d'emoji. Un score obtenu dessus
  est donc optimiste par rapport à de vrais messages.
- **850 items donnent un intervalle de confiance de 3,4 points** à 95 % sur
  l'exactitude phrase. Deux modèles séparés par moins que cela ne sont pas
  distinguables. Le CER, mesuré au caractère, est plus fiable.
- **163 items sont des mots isolés** et non des phrases. La colonne `mot_isole`
  permet de les évaluer à part.

```{important}
Un jeu de test n'est verrouillé que si l'on cesse de le regarder. Les mesures de
mise au point se font sur `dev.csv`. Le jeu de test ne s'ouvre qu'aux jalons
décidés à l'avance, et chaque ouverture est consignée dans {doc}`../changelog`
avec sa date et son objet.
```
