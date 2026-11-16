# Validation humaine et registre d'erreurs

Conception arrêtée le 16 novembre 2026, avant implémentation.

## Ce qui est demandé

La directive *Score de confiance et boucle de feedback* du cadrage du projet
impose trois choses à tout modèle livré :

1. être packagé dans des **conteneurs isolés, avec une base de données
   dédiée**, pour collecter automatiquement les erreurs ;
2. faire passer **toute prédiction** par une **interface de validation
   humaine**, où l'utilisateur confirme ou corrige ;
3. sauvegarder automatiquement cette validation dans un **registre d'erreurs**,
   jeu de réentraînement qui alimente le fine-tuning.

Le score de confiance, déjà renvoyé par le service, ne dispense pas de cette
validation : il sert à l'ordonner.

## Ce qui n'est pas fait

Pas de campagne d'annotation : ni recrutement d'annotateurs, ni double
annotation, ni constitution d'un jeu de test. Les validateurs sont les
utilisateurs du système, qui vérifient ses propres sorties. Pas d'hébergement
public non plus : l'ensemble tourne en local, par `docker compose`.

## Architecture

```{mermaid}
flowchart LR
    C[Client : Sawti, Swagger, script] -->|POST /normaliser| A
    subgraph compose[docker compose]
        A["api<br/>ByT5 + calibration"] -->|enregistre chaque prédiction| B[("base<br/>Postgres")]
        V["validation<br/>pages HTML"] -->|lit, enregistre les décisions| B
    end
    H[Validateur] -->|confirme ou corrige| V
    B -->|python -m registre.exporter| R["registre.csv<br/>transliteration, darija_norm"]
    R -->|TRAIN_CSV| T[modele/train_byt5.py]
```

Trois conteneurs, chacun avec un seul rôle :

| Conteneur | Rôle | Dépend du modèle |
|---|---|---|
| `api` | le service existant ; enregistre en plus chaque prédiction | oui |
| `base` | Postgres, avec un volume : les données survivent au redémarrage | non |
| `validation` | les pages de validation | non |

Le code commun aux deux conteneurs applicatifs, le schéma de la base et
l'export, vit dans un paquet `registre/`. L'interface est un paquet
`validation/` sans torch ni modèle.

## Données

```
predictions                          validations
-----------                          -----------
id                                   id
cree_le                              prediction_id  -> predictions.id
entree        texte reçu, tel quel   cree_le
entree_propre texte vu par le modèle decision       confirmee | corrigee | ecartee
sortie                               correction     texte retenu, vide si écartée
confiance     rang calibré           motif          si écartée
confiance_brute                      validateur     nom saisi une fois
complete                             remarque
revision      du modèle
statut        a_valider | confirmee | corrigee | ecartee
```

Les validations **ne sont jamais écrasées** : revenir sur une phrase ajoute une
ligne, la plus récente fait foi, et `predictions.statut` la reflète. Une vue
`registre_erreurs` donne, pour chaque prédiction corrigée, l'entrée, la sortie
du modèle, la correction et la confiance : c'est le registre interrogeable en
SQL.

## Enregistrement par le service

- Si `DATABASE_URL` est définie, chaque prédiction, seule ou dans un lot, est
  enregistrée **avant** que la réponse parte, et la réponse porte son `id`.
- Si la base ne répond pas, le service renvoie **503** au lieu d'une
  prédiction non enregistrée : la directive exige que toute prédiction passe
  par la validation, et une prédiction perdue y échapperait sans que rien ne
  le signale.
- Sans `DATABASE_URL`, le service fonctionne comme aujourd'hui, sans
  enregistrer : c'est le cas des tests et d'un usage hors de la chaîne.

## Interface de validation

- **La liste** montre les prédictions à valider, **les moins sûres d'abord** :
  entrée, sortie et rang de confiance, par pages de vingt.
- **Une prédiction ouverte** affiche l'arabizi et un champ de droite à gauche
  pré-rempli avec la sortie du modèle, plus des boutons pour ڭ ڤ ؟ ،.
  - *Confirmer* : la sortie est juste.
  - *Corriger* : le texte modifié est enregistré comme correction.
  - *Écarter*, avec un motif : pas du darija, entrée inutilisable, donnée
    personnelle. Une prédiction écartée n'entre jamais dans le registre.
- **L'historique** liste les décisions prises ; chacune se rouvre et se
  modifie. Une correction est visible après rechargement de la page.
- Le validateur saisit son nom une fois, gardé dans un cookie. Il n'y a pas
  d'authentification : l'outil tourne en local.

Une correction contenant de la vocalisation est refusée avec un message, la
convention de sortie étant sans vocalisation. Rien n'est corrigé à la place du
validateur.

## Export pour le réentraînement

`python -m registre.exporter registre.csv` écrit les prédictions corrigées avec
les colonnes que lit `modele/train_byt5.py` sans transformation :
`transliteration` (l'entrée vue par le modèle) et `darija_norm` (la correction
normalisée comme les autres cibles, par `donnees/make_splits.py`). Il ajoute
`darija`, `sortie_modele`, `confiance` et `revision` pour l'analyse.

- `--avec-confirmations` ajoute les sorties confirmées, comme exemples justes.
- **Toute phrase du jeu de test est retirée de l'export**, avec la même clé que
  le découpage du corpus, et le nombre de lignes retirées est affiché. Sans ce
  garde-fou, une phrase de TerjamaBench soumise au service puis corrigée
  entrerait dans l'entraînement, et l'évaluation serait faussée.

## Tests

Sur SQLite, sans serveur Postgres ni modèle :

- chaque appel au service enregistre une prédiction, un lot en enregistre *n*,
  et l'`id` renvoyé est le bon ;
- base indisponible : 503 ; sans `DATABASE_URL` : aucun enregistrement ;
- la liste est triée par confiance croissante ;
- confirmer, corriger, écarter mettent le statut à jour ; une correction est
  visible après rechargement ; écarter sans motif est refusé ;
- l'historique garde chaque décision ;
- l'export a les colonnes attendues, normalise la cible, exclut les écartées,
  et retire les phrases du jeu de test.

La persistance au redémarrage se vérifie avec `docker compose` : redémarrer les
conteneurs, retrouver les validations.
