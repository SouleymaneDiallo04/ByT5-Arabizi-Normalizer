# Validation humaine et registre d'erreurs : plan d'implémentation

> **Pour les agents :** sous-compétence requise : superpowers:subagent-driven-development
> (recommandée) ou superpowers:executing-plans, tâche par tâche. Les étapes se
> cochent (`- [ ]`).

**Objectif.** Toute prédiction du service est enregistrée dans une base
dédiée. Une interface la fait confirmer, corriger ou écarter par un humain, et
les corrections s'exportent en jeu de réentraînement.

**Architecture.** Un paquet `registre/` (schéma SQLAlchemy Core, accès, export)
est partagé par le service, qui écrit les prédictions, et par l'interface
`validation/` (FastAPI + Jinja2, sans torch), qui écrit les décisions.
`docker compose` assemble trois conteneurs : `api`, `base` (Postgres, volume
persistant) et `validation`. Conception : `docs/project/validation-humaine.md`.

**Technologies.** Python 3.13, FastAPI 0.136.1, SQLAlchemy 2.0.51 (Core, pas
d'ORM), psycopg 3.3.9, Jinja2 3.1.6, python-multipart 0.0.26, Postgres 17.12,
pytest sur SQLite.

## Contraintes globales

- Directive du professeur (`5A-Expertise.xlsx`, feuille Directives) : **toute**
  prédiction passe par la validation, et les validations alimentent un registre
  d'erreurs réutilisable pour le fine-tuning.
- Les validations **ne sont jamais écrasées ni supprimées**. La plus récente
  fait foi, et `predictions.statut` la reflète.
- Si la base ne répond pas, le service renvoie **503**, et non une prédiction
  non enregistrée. Sans `DATABASE_URL`, il n'enregistre rien, comme aujourd'hui.
- Pas d'authentification. Les deux ports ne sont publiés que sur `127.0.0.1`.
- Une correction vocalisée est **refusée avec un message**. Rien n'est corrigé à
  la place du validateur.
- L'export retire toute phrase du jeu de test, avec les clés de
  `donnees/make_splits.py`, et affiche combien de lignes il a retirées.
- Versions figées (ENF-04) : `sqlalchemy==2.0.51`, `psycopg[binary]==3.3.9`,
  `jinja2==3.1.6`, `python-multipart==0.0.26`, `postgres:17.12-alpine`.
  Postgres 18 a déplacé son répertoire de données : son volume ne se monte plus
  au même chemin.
- Code, noms et messages en français, sans attribution d'outil dans les
  commits. Tests lancés depuis la racine : `py -m pytest -q` (30 tests passent
  au départ).
- La branche `validation-humaine` est empilée sur `metriques-wer` (PR #23) : sa
  PR se fusionne après celle-ci.

## Fichiers

| Fichier | Rôle |
|---|---|
| `donnees/normalisation.py` (nouveau) | `normaliser_cible`, `cle_arabe`, `cle_latine`, `contient_vocalisation`, sans pandas |
| `donnees/make_splits.py` | réexporte ces fonctions, sans les redéfinir |
| `registre/schema.py` (nouveau) | tables `predictions` et `validations`, vue `registre_erreurs`, `creer_tables` |
| `registre/depot.py` (nouveau) | classe `Registre` : enregistrer, lister, valider, historique, à exporter |
| `registre/exporter.py` (nouveau) | `python -m registre.exporter registre.csv` |
| `service/config.py`, `service/api.py` | `DATABASE_URL`, enregistrement, `id` dans la réponse, 503 |
| `validation/app.py`, `validation/gabarits/*.html` (nouveaux) | les pages de validation |
| `validation/Dockerfile`, `compose.yaml`, `requirements-validation.txt` (nouveaux) | les conteneurs |
| `Dockerfile`, `.dockerignore`, `requirements-service.txt` | le service embarque le registre |
| `README.md`, `docs/project/validation-humaine.md`, `docs/project/decisions.md` | mode d'emploi et journal |

---

### Tâche 1 : normalisation sans pandas

Le registre et l'interface ont besoin de `normaliser_cible` et des clés, mais
`make_splits.py` importe pandas et huggingface_hub dès la première ligne.
L'image de validation n'en a que faire. Les fonctions passent dans un module
sans dépendance, que `make_splits.py` réexporte : `melanger.py`,
`ajouter_source.py` et leurs tests continuent d'importer depuis `make_splits`.

**Fichiers :**
- Créer : `donnees/normalisation.py`
- Modifier : `donnees/make_splits.py:1-50`
- Test : `tests/test_normalisation.py`

**Interfaces :**
- Produit : `normaliser_cible(texte) -> str`, `cle_arabe(texte) -> str`,
  `cle_latine(texte) -> str`, `contient_vocalisation(texte) -> bool`, toutes
  dans `donnees.normalisation`.

- [ ] **Étape 1 : écrire le test qui échoue**

`tests/test_normalisation.py` :

```python
"""Normalisation partagee par le decoupage, le registre et l'interface."""

import subprocess
import sys
from pathlib import Path

from donnees import make_splits, normalisation

RACINE = Path(__file__).resolve().parent.parent


def test_make_splits_reexporte_les_memes_fonctions():
    assert make_splits.normaliser_cible is normalisation.normaliser_cible
    assert make_splits.cle_arabe is normalisation.cle_arabe
    assert make_splits.cle_latine is normalisation.cle_latine


def test_normaliser_cible():
    assert normalisation.normaliser_cible("واش  كاين?") == "واش كاين؟"
    assert normalisation.normaliser_cible("شْريت") == "شريت"


def test_cles():
    assert normalisation.cle_arabe("أنا، هنا!") == "انا هنا"
    assert normalisation.cle_latine("Ana, HNA!") == "ana hna"


def test_contient_vocalisation():
    assert normalisation.contient_vocalisation("شْريت")
    assert normalisation.contient_vocalisation("كـبير")
    assert not normalisation.contient_vocalisation("شريت ڭاع، واش؟")


def test_sans_pandas():
    """L'image de validation n'a ni pandas ni huggingface_hub."""
    code = ("import sys, donnees.normalisation; "
            "print('pandas' in sys.modules, 'huggingface_hub' in sys.modules)")
    sortie = subprocess.run([sys.executable, "-c", code], cwd=RACINE,
                            capture_output=True, text=True, check=True)
    assert sortie.stdout.split() == ["False", "False"]
```

- [ ] **Étape 2 : vérifier qu'il échoue**

Lancer : `py -m pytest tests/test_normalisation.py -q`
Attendu : ÉCHEC, `ImportError: cannot import name 'normalisation'`.

- [ ] **Étape 3 : créer le module**

`donnees/normalisation.py` : les lignes 15 à 50 de `make_splits.py`, avec leurs
commentaires, déplacées sans modification, plus `contient_vocalisation`.

```python
"""Normalisation des textes et cles de rapprochement, sans dependance.

Sortie de make_splits.py pour que le registre et l'interface de validation s'en
servent sans charger pandas ni huggingface_hub. make_splits.py les reexporte :
ses appelants n'ont rien a changer.
"""

import re
import unicodedata

# Signes de vocalisation et tatweel : TerjamaBench n'en porte pas (0,2 % des mots),
# alors que doda_synthetic en porte 21 %. Les laisser ferait compter chaque
# diacritique produit comme une erreur de caractère, sans erreur de langue.
VOCALISATION = [chr(c) for c in range(0x064B, 0x0656)] + ["ـ", "ٰ"]
TABLE_VOCAL = {ord(c): None for c in VOCALISATION}
# TerjamaBench écrit le point d'interrogation en arabe dans 100 % des cas,
# DODa-10K en latin dans 97 % des cas.
PONCTUATION_ARABE = {ord("?"): "؟", ord(";"): "؛"}

PONCT_CLE = (
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "،؛؟٪٫٬٭۔"
    "«»‘’“”…–—"
)
TABLE_PONCT = {ord(c): " " for c in PONCT_CLE}
EQUIVALENCES = [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ة", "ه"), ("ى", "ي")]


def normaliser_cible(texte):
    # Ce que le modèle devra produire. Seules les différences non débattues sont
    # corrigées ici ; la convention orthographique proprement dite reste à arbitrer.
    t = unicodedata.normalize("NFKC", str(texte)).translate(TABLE_VOCAL)
    return re.sub(r"\s+", " ", t.translate(PONCTUATION_ARABE)).strip()


def cle_arabe(texte):
    t = unicodedata.normalize("NFKC", str(texte)).translate(TABLE_PONCT).translate(TABLE_VOCAL)
    for avant, apres in EQUIVALENCES:
        t = t.replace(avant, apres)
    return re.sub(r"\s+", " ", t).strip()


def cle_latine(texte):
    t = unicodedata.normalize("NFKC", str(texte)).lower().translate(TABLE_PONCT)
    return re.sub(r"\s+", " ", t).strip()


def contient_vocalisation(texte):
    """Vrai si le texte porte un signe que normaliser_cible retirerait."""
    return any(ord(c) in TABLE_VOCAL for c in str(texte))
```

- [ ] **Étape 4 : faire réexporter `make_splits.py`**

Dans `donnees/make_splits.py`, supprimer `import unicodedata` (ligne 3, qui ne
sert plus après le déplacement ; `re` sert encore à la ligne 63) et remplacer
les lignes 15 à 50, des commentaires de `VOCALISATION` jusqu'à la fin de
`cle_latine`, par :

```python
# Definies dans normalisation.py, sans dependance, pour le registre et
# l'interface de validation. Reexportees ici : melanger.py, ajouter_source.py
# et les tests les importent depuis ce module.
from donnees.normalisation import (  # noqa: F401
    EQUIVALENCES, PONCT_CLE, PONCTUATION_ARABE, TABLE_PONCT, TABLE_VOCAL,
    VOCALISATION, cle_arabe, cle_latine, normaliser_cible)
```

- [ ] **Étape 5 : vérifier**

Lancer : `py -m pytest -q`
Attendu : 35 réussis (30 + 5), dont `tests/test_ajouter_source.py`, qui
importe depuis `make_splits`.

- [ ] **Étape 6 : commit**

```bash
git add donnees/normalisation.py donnees/make_splits.py tests/test_normalisation.py
git commit -m "Normalisation dans un module sans pandas, reexportee par make_splits"
```

---

### Tâche 2 : le registre

**Fichiers :**
- Créer : `registre/__init__.py`, `registre/schema.py`, `registre/depot.py`
- Test : `tests/test_registre.py`

**Interfaces :**
- Produit, dans `registre.schema` : `DECISIONS = ("confirmee", "corrigee", "ecartee")`,
  `STATUTS`, `MOTIFS` (dict code → libellé), tables `predictions` et
  `validations`, `creer_tables(moteur)`.
- Produit, dans `registre.depot`, la classe `Registre(url)`, avec l'attribut
  `moteur` et les méthodes :
  - `creer_tables() -> None` ;
  - `enregistrer(lignes: list[dict]) -> list[int]`. Chaque dict porte `entree`,
    `entree_propre`, `sortie`, `confiance`, `confiance_brute`, `complete` et
    `revision`. Les id sont rendus dans l'ordre des lignes.
  - `a_valider(page=1, par_page=20) -> (list[dict], int)`, les moins sûres
    d'abord ;
  - `premiere_a_valider() -> int | None` ;
  - `prediction(id) -> dict | None`. La clé `derniere` contient la dernière
    validation, ou None.
  - `valider(id, decision, validateur, correction=None, motif=None, remarque=None) -> int`.
    Lève `ValueError` sur une décision incomplète, et `LookupError` sur un id
    inconnu.
  - `historique(page=1, par_page=20) -> (list[dict], int)`, la plus récente
    d'abord ;
  - `a_exporter(avec_confirmations=False) -> list[dict]`, avec en plus la clé
    `darija`, le texte retenu.

- [ ] **Étape 1 : écrire les tests qui échouent**

`tests/test_registre.py` :

```python
"""Tests du registre, sur SQLite : ni serveur Postgres ni modele."""

import pytest
from sqlalchemy import text

from registre.depot import Registre


def ligne(entree, confiance, sortie=None):
    return {"entree": entree, "entree_propre": entree.strip(),
            "sortie": sortie or f"<{entree.strip()}>", "confiance": confiance,
            "confiance_brute": 0.95, "complete": True, "revision": "rev-test"}


@pytest.fixture
def registre(tmp_path):
    r = Registre(f"sqlite:///{(tmp_path / 'registre.db').as_posix()}")
    r.creer_tables()
    return r


def test_enregistrer_rend_les_id_dans_l_ordre(registre):
    ids = registre.enregistrer([ligne("a", 0.5), ligne("b", 0.2), ligne("c", 0.9)])
    assert len(set(ids)) == 3
    assert [registre.prediction(i)["entree"] for i in ids] == ["a", "b", "c"]
    assert {registre.prediction(i)["statut"] for i in ids} == {"a_valider"}
    assert registre.prediction(ids[0])["derniere"] is None


def test_creer_tables_deux_fois(registre):
    registre.creer_tables()


def test_a_valider_trie_par_confiance_croissante(registre):
    ids = registre.enregistrer([ligne("a", 0.5), ligne("b", 0.2), ligne("c", 0.9)])
    registre.valider(ids[1], "confirmee", "Awa")
    lignes, total = registre.a_valider()
    assert [x["entree"] for x in lignes] == ["a", "c"] and total == 2
    assert registre.premiere_a_valider() == ids[0]


def test_a_valider_par_pages(registre):
    registre.enregistrer([ligne(f"p{i:02d}", i / 100) for i in range(25)])
    page1, total = registre.a_valider(1, 20)
    page2, _ = registre.a_valider(2, 20)
    assert total == 25 and len(page1) == 20 and len(page2) == 5
    assert page2[0]["entree"] == "p20"


def test_rien_a_valider(registre):
    assert registre.a_valider() == ([], 0)
    assert registre.premiere_a_valider() is None


def test_valider_met_le_statut_a_jour_et_garde_l_historique(registre):
    [i] = registre.enregistrer([ligne("chrit", 0.3)])
    registre.valider(i, "corrigee", "Awa", correction="شريت")
    registre.valider(i, "confirmee", "Hinimdou")
    p = registre.prediction(i)
    assert p["statut"] == "confirmee"
    assert p["derniere"]["validateur"] == "Hinimdou"
    assert p["derniere"]["correction"] is None
    lignes, total = registre.historique()
    assert total == 2
    assert [x["decision"] for x in lignes] == ["confirmee", "corrigee"]
    assert lignes[1]["correction"] == "شريت" and lignes[1]["entree"] == "chrit"


@pytest.mark.parametrize("decision, options, message", [
    ("validee", {}, "inconnue"),
    ("corrigee", {"correction": "  "}, "correction"),
    ("ecartee", {}, "motif"),
    ("ecartee", {"motif": "autre"}, "motif"),
])
def test_valider_refuse_l_incomplet(registre, decision, options, message):
    [i] = registre.enregistrer([ligne("chrit", 0.3)])
    with pytest.raises(ValueError, match=message):
        registre.valider(i, decision, "Awa", **options)
    assert registre.prediction(i)["statut"] == "a_valider"
    assert registre.historique()[1] == 0


def test_valider_sans_nom_refuse(registre):
    [i] = registre.enregistrer([ligne("chrit", 0.3)])
    with pytest.raises(ValueError, match="validateur"):
        registre.valider(i, "confirmee", "  ")


def test_prediction_inconnue(registre):
    assert registre.prediction(999) is None
    with pytest.raises(LookupError):
        registre.valider(999, "confirmee", "Awa")


def test_vue_registre_erreurs(registre):
    a, b, c = registre.enregistrer([ligne("a", 0.1), ligne("b", 0.2), ligne("c", 0.3)])
    registre.valider(a, "corrigee", "Awa", correction="أ")
    registre.valider(a, "corrigee", "Awa", correction="ا")
    registre.valider(b, "confirmee", "Awa")
    registre.valider(c, "corrigee", "Awa", correction="س")
    registre.valider(c, "ecartee", "Awa", motif="pas_darija")
    with registre.moteur.connect() as conn:
        lignes = conn.execute(text(
            "SELECT prediction_id, correction FROM registre_erreurs")).all()
    assert [tuple(x) for x in lignes] == [(a, "ا")]


def test_a_exporter_suit_la_derniere_decision(registre):
    a, b, c, d = registre.enregistrer(
        [ligne("a", 0.1), ligne("b", 0.2), ligne("c", 0.3), ligne("d", 0.4)])
    registre.valider(a, "corrigee", "Awa", correction="un")
    registre.valider(a, "corrigee", "Awa", correction="deux")
    registre.valider(b, "confirmee", "Awa")
    registre.valider(c, "ecartee", "Awa", motif="donnee_personnelle")
    # d reste a valider : jamais exportee
    assert [(x["id"], x["darija"]) for x in registre.a_exporter()] == [(a, "deux")]
    avec = registre.a_exporter(avec_confirmations=True)
    assert [(x["id"], x["darija"]) for x in avec] == [(a, "deux"), (b, "<b>")]
```

- [ ] **Étape 2 : vérifier qu'ils échouent**

Lancer : `py -m pytest tests/test_registre.py -q`
Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'registre'`.

- [ ] **Étape 3 : écrire le schéma**

`registre/__init__.py` :

```python
"""Registre des predictions et de leur validation humaine."""
```

`registre/schema.py` :

```python
"""Schema du registre : les predictions du service et leurs validations.

Les validations ne sont jamais modifiees ni supprimees. Revenir sur une phrase
ajoute une ligne ; la plus recente fait foi, et predictions.statut la reflete.
"""

from sqlalchemy import (Boolean, CheckConstraint, Column, DateTime, Float,
                        ForeignKey, Index, Integer, MetaData, String, Table,
                        Text, func, text)

DECISIONS = ("confirmee", "corrigee", "ecartee")
STATUTS = ("a_valider",) + DECISIONS
# Codes stockes en base, libelles affiches par l'interface.
MOTIFS = {
    "pas_darija": "pas du darija",
    "entree_inutilisable": "entrée inutilisable",
    "donnee_personnelle": "donnée personnelle",
}


def _parmi(colonne, valeurs):
    return f"{colonne} IN ({', '.join(repr(v) for v in valeurs)})"


metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True),
    Column("cree_le", DateTime(timezone=True), nullable=False,
           server_default=func.now()),
    Column("entree", Text, nullable=False),          # le texte recu, tel quel
    Column("entree_propre", Text, nullable=False),   # le texte vu par le modele
    Column("sortie", Text, nullable=False),
    Column("confiance", Float, nullable=False),      # rang calibre (PO-05)
    Column("confiance_brute", Float, nullable=False),
    Column("complete", Boolean, nullable=False),
    Column("revision", Text, nullable=False),        # du modele servi
    Column("statut", String(16), nullable=False, server_default="a_valider"),
    CheckConstraint(_parmi("statut", STATUTS), name="statut_connu"),
    Index("a_valider_par_confiance", "statut", "confiance"),
)

validations = Table(
    "validations", metadata,
    Column("id", Integer, primary_key=True),
    Column("prediction_id", Integer, ForeignKey("predictions.id"),
           nullable=False, index=True),
    Column("cree_le", DateTime(timezone=True), nullable=False,
           server_default=func.now()),
    Column("decision", String(16), nullable=False),
    Column("correction", Text),       # le texte retenu, si corrigee
    Column("motif", String(32)),      # si ecartee
    Column("validateur", Text, nullable=False),
    Column("remarque", Text),
    CheckConstraint(_parmi("decision", DECISIONS), name="decision_connue"),
)

# Le registre d'erreurs interrogeable en SQL : chaque prediction corrigee, avec
# sa derniere correction. L'alias w est necessaire : sans lui, la sous-requete
# se correlerait a la jointure et rendrait chaque validation.
VUE_ERREURS = """
SELECT p.id AS prediction_id, p.entree, p.entree_propre, p.sortie,
       v.correction, p.confiance, p.revision, v.validateur,
       v.cree_le AS corrige_le
FROM predictions p
JOIN validations v
  ON v.id = (SELECT max(w.id) FROM validations w WHERE w.prediction_id = p.id)
WHERE p.statut = 'corrigee'
"""


def creer_tables(moteur):
    """Cree tables et vue si elles manquent ; sans effet sur une base existante."""
    metadata.create_all(moteur)
    # Postgres ne connait pas CREATE VIEW IF NOT EXISTS, SQLite pas OR REPLACE.
    entete = ("CREATE OR REPLACE VIEW" if moteur.dialect.name == "postgresql"
              else "CREATE VIEW IF NOT EXISTS")
    with moteur.begin() as conn:
        conn.execute(text(f"{entete} registre_erreurs AS {VUE_ERREURS}"))
```

- [ ] **Étape 4 : écrire l'accès**

`registre/depot.py` :

```python
"""Acces au registre : le service y ecrit les predictions, l'interface les
decisions. SQLAlchemy Core, sans ORM : deux tables n'en demandent pas plus."""

from sqlalchemy import create_engine, func, insert, select, update

from registre.schema import (DECISIONS, MOTIFS, creer_tables, predictions,
                             validations)

CHAMPS = ("entree", "entree_propre", "sortie", "confiance", "confiance_brute",
          "complete", "revision")


class Registre:
    def __init__(self, url):
        options = {"pool_pre_ping": True}
        if url.startswith("postgresql"):
            # Base arretee : echouer en cinq secondes, pour que le service
            # renvoie 503, plutot que de laisser la requete pendre.
            options["connect_args"] = {"connect_timeout": 5}
        self.moteur = create_engine(url, **options)

    def creer_tables(self):
        creer_tables(self.moteur)

    def enregistrer(self, lignes):
        """Enregistre des predictions en une transaction ; rend leurs id dans l'ordre."""
        ids = []
        with self.moteur.begin() as conn:
            for ligne in lignes:
                valeurs = {champ: ligne[champ] for champ in CHAMPS}
                resultat = conn.execute(insert(predictions).values(**valeurs))
                ids.append(resultat.inserted_primary_key[0])
        return ids

    def a_valider(self, page=1, par_page=20):
        """Predictions en attente, les moins sures d'abord : (lignes, total)."""
        attente = predictions.c.statut == "a_valider"
        with self.moteur.connect() as conn:
            total = conn.execute(select(func.count()).select_from(predictions)
                                 .where(attente)).scalar_one()
            lignes = conn.execute(
                select(predictions).where(attente)
                .order_by(predictions.c.confiance, predictions.c.id)
                .limit(par_page).offset((page - 1) * par_page)).mappings().all()
        return [dict(x) for x in lignes], total

    def premiere_a_valider(self):
        lignes, _ = self.a_valider(1, 1)
        return lignes[0]["id"] if lignes else None

    def prediction(self, prediction_id):
        """La prediction et sa derniere validation (cle derniere), ou None."""
        with self.moteur.connect() as conn:
            p = conn.execute(select(predictions).where(
                predictions.c.id == prediction_id)).mappings().first()
            if p is None:
                return None
            v = conn.execute(
                select(validations)
                .where(validations.c.prediction_id == prediction_id)
                .order_by(validations.c.id.desc()).limit(1)).mappings().first()
        return {**p, "derniere": dict(v) if v else None}

    def valider(self, prediction_id, decision, validateur, correction=None,
                motif=None, remarque=None):
        """Ajoute une validation et met le statut a jour, en une transaction."""
        if decision not in DECISIONS:
            raise ValueError(f"décision inconnue : {decision}")
        if not (validateur or "").strip():
            raise ValueError("validateur manquant")
        if decision == "corrigee" and not (correction or "").strip():
            raise ValueError("une correction est requise")
        if decision == "ecartee" and motif not in MOTIFS:
            raise ValueError("un motif est requis pour écarter")
        valeurs = {
            "prediction_id": prediction_id, "decision": decision,
            "correction": correction if decision == "corrigee" else None,
            "motif": motif if decision == "ecartee" else None,
            "validateur": validateur.strip(),
            "remarque": (remarque or "").strip() or None,
        }
        with self.moteur.begin() as conn:
            existe = conn.execute(select(predictions.c.id).where(
                predictions.c.id == prediction_id)).first()
            if existe is None:
                raise LookupError(f"prédiction {prediction_id} inconnue")
            resultat = conn.execute(insert(validations).values(**valeurs))
            conn.execute(update(predictions)
                         .where(predictions.c.id == prediction_id)
                         .values(statut=decision))
        return resultat.inserted_primary_key[0]

    def historique(self, page=1, par_page=20):
        """Toutes les decisions, les plus recentes d'abord : (lignes, total)."""
        with self.moteur.connect() as conn:
            total = conn.execute(select(func.count())
                                 .select_from(validations)).scalar_one()
            lignes = conn.execute(
                select(validations, predictions.c.entree, predictions.c.sortie,
                       predictions.c.statut)
                .join(predictions, predictions.c.id == validations.c.prediction_id)
                .order_by(validations.c.id.desc())
                .limit(par_page).offset((page - 1) * par_page)).mappings().all()
        return [dict(x) for x in lignes], total

    def a_exporter(self, avec_confirmations=False):
        """Predictions dont la derniere decision est une correction, ou une
        confirmation sur demande. La cle darija porte le texte retenu."""
        statuts = ["corrigee"] + (["confirmee"] if avec_confirmations else [])
        # Alias : sans lui, la sous-requete se correlerait a la jointure.
        w = validations.alias("w")
        derniere = (select(func.max(w.c.id))
                    .where(w.c.prediction_id == predictions.c.id)
                    .scalar_subquery())
        requete = (select(predictions, validations.c.correction)
                   .join(validations, validations.c.id == derniere)
                   .where(predictions.c.statut.in_(statuts))
                   .order_by(predictions.c.id))
        with self.moteur.connect() as conn:
            lignes = conn.execute(requete).mappings().all()
        return [{**x, "darija": x["correction"] if x["statut"] == "corrigee"
                 else x["sortie"]} for x in lignes]
```

- [ ] **Étape 5 : vérifier**

Lancer : `py -m pytest tests/test_registre.py -q`
Attendu : 14 réussis. Si `test_a_exporter_suit_la_derniere_decision` rend
deux lignes pour `a`, c'est que la sous-requête a été corrélée à la jointure :
vérifier l'alias `w`.

- [ ] **Étape 6 : commit**

```bash
git add registre/ tests/test_registre.py
git commit -m "Registre des predictions et des validations, sur SQLAlchemy Core"
```

---

### Tâche 3 : l'export pour le réentraînement

**Fichiers :**
- Créer : `registre/exporter.py`
- Test : `tests/test_exporter.py`

**Interfaces :**
- Consomme : `Registre.a_exporter`, ainsi que `cle_arabe`, `cle_latine` et
  `normaliser_cible` de `donnees.normalisation`.
- Produit : `COLONNES` ; `exporter(registre, sortie, test="test_real.csv",
  avec_confirmations=False) -> dict`, qui rend le bilan avec les clés
  `corrigees`, `confirmees`, `retirees_test` et `ecrites` ; `main(arguments=None) -> dict`.

- [ ] **Étape 1 : écrire les tests qui échouent**

`tests/test_exporter.py` :

```python
"""Export du registre : colonnes, cible normalisee, garde-fou du jeu de test."""

import csv

import pytest

from registre.depot import Registre
from registre.exporter import COLONNES, exporter, main


def ligne(entree, sortie):
    return {"entree": f"  {entree} ", "entree_propre": entree, "sortie": sortie,
            "confiance": 0.42, "confiance_brute": 0.95, "complete": True,
            "revision": "rev-test"}


@pytest.fixture
def registre(tmp_path):
    r = Registre(f"sqlite:///{(tmp_path / 'registre.db').as_posix()}")
    r.creer_tables()
    return r


@pytest.fixture
def jeu_de_test(tmp_path):
    chemin = tmp_path / "test_real.csv"
    with open(chemin, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "darija", "transliteration"])
        w.writerow(["terjama_0000", "الله يسعد الماسا", "lays3d lmasa"])
    return chemin


def lire(chemin):
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_colonnes_et_cible_normalisee(registre, jeu_de_test, tmp_path):
    [i] = registre.enregistrer([ligne("wach kayn", "واش كاين")])
    registre.valider(i, "corrigee", "Awa", correction="واش كاين?")
    sortie = tmp_path / "registre.csv"
    bilan = exporter(registre, sortie, jeu_de_test)
    [x] = lire(sortie)
    assert list(x) == COLONNES
    assert x["transliteration"] == "wach kayn"      # l'entree vue par le modele
    assert x["darija"] == "واش كاين?"
    assert x["darija_norm"] == "واش كاين؟"
    assert x["sortie_modele"] == "واش كاين"
    assert x["id"] == f"registre_{i:06d}" and x["source"] == "registre"
    assert x["revision"] == "rev-test" and float(x["confiance"]) == 0.42
    assert bilan == {"corrigees": 1, "confirmees": 0, "retirees_test": 0,
                     "ecrites": 1}


def test_ecartees_exclues_confirmations_sur_demande(registre, jeu_de_test, tmp_path):
    a, b, c, _ = registre.enregistrer([ligne("a1", "ا1"), ligne("b1", "ب1"),
                                       ligne("c1", "ج1"), ligne("d1", "د1")])
    registre.valider(a, "corrigee", "Awa", correction="ا2")
    registre.valider(b, "confirmee", "Awa")
    registre.valider(c, "ecartee", "Awa", motif="pas_darija")
    sortie = tmp_path / "registre.csv"
    exporter(registre, sortie, jeu_de_test)
    assert [x["transliteration"] for x in lire(sortie)] == ["a1"]
    bilan = exporter(registre, sortie, jeu_de_test, avec_confirmations=True)
    lignes = lire(sortie)
    assert [(x["transliteration"], x["darija"]) for x in lignes] == [
        ("a1", "ا2"), ("b1", "ب1")]
    assert bilan["confirmees"] == 1 and bilan["ecrites"] == 2


def test_phrases_du_test_retirees(registre, jeu_de_test, tmp_path):
    a, b, c = registre.enregistrer([
        ligne("Lays3d lmasa!", "اللاه يسعد"),      # meme arabizi, casse et ponctuation a part
        ligne("autre chose", "الله يسعد"),          # meme arabe, une fois corrige
        ligne("chrit", "شرط")])
    registre.valider(a, "corrigee", "Awa", correction="الله يسعد المسا")
    registre.valider(b, "corrigee", "Awa", correction="الله يسعد الماسا.")
    registre.valider(c, "corrigee", "Awa", correction="شريت")
    sortie = tmp_path / "registre.csv"
    bilan = exporter(registre, sortie, jeu_de_test)
    assert [x["transliteration"] for x in lire(sortie)] == ["chrit"]
    assert bilan["retirees_test"] == 2 and bilan["ecrites"] == 1


def test_jeu_de_test_absent(registre, tmp_path):
    with pytest.raises(SystemExit, match="make_splits"):
        exporter(registre, tmp_path / "x.csv", tmp_path / "absent.csv")


def test_ligne_de_commande(registre, jeu_de_test, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", str(registre.moteur.url))
    [i] = registre.enregistrer([ligne("chrit", "شريت")])
    registre.valider(i, "confirmee", "Awa")
    bilan = main([str(tmp_path / "r.csv"), "--test", str(jeu_de_test),
                  "--avec-confirmations"])
    assert bilan["ecrites"] == 1
    assert "écrites" in capsys.readouterr().out


def test_ligne_de_commande_sans_base(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit, match="DATABASE_URL"):
        main([str(tmp_path / "r.csv")])
```

- [ ] **Étape 2 : vérifier qu'ils échouent**

Lancer : `py -m pytest tests/test_exporter.py -q`
Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'registre.exporter'`.

- [ ] **Étape 3 : écrire l'export**

`registre/exporter.py` :

```python
"""Export du registre pour le reentrainement.

Usage, la base designee par DATABASE_URL :
    python -m registre.exporter registre.csv
    python -m registre.exporter registre.csv --avec-confirmations --test test_real.csv

transliteration et darija_norm sont les colonnes que lit modele/train_byt5.py ;
id, darija et source celles des autres corpus, pour que le fichier se melange
comme eux. sortie_modele, confiance et revision servent a l'analyse.

ATTENTION. Toute phrase du jeu de test est retiree, avec les cles de
make_splits.py. Sans ce garde-fou, une phrase de TerjamaBench soumise au
service puis corrigee entrerait dans l'entrainement, et l'evaluation ne
vaudrait plus rien.
"""

import argparse
import csv
import os

from donnees.normalisation import cle_arabe, cle_latine, normaliser_cible
from registre.depot import Registre

COLONNES = ["id", "transliteration", "darija_norm", "darija", "source",
            "sortie_modele", "confiance", "revision"]
TEST = "test_real.csv"


def cles_du_test(chemin):
    try:
        with open(chemin, encoding="utf-8-sig", newline="") as f:
            lignes = list(csv.DictReader(f))
    except FileNotFoundError:
        raise SystemExit(
            f"{chemin} absent : sans lui, rien ne garantit que l'export "
            "epargne le jeu de test. Le produire par python -m "
            "donnees.make_splits, ou le telecharger :\n"
            "  hf download sawti-nlp/darija-arabizi-corpus splits/test_real.csv "
            "--repo-type dataset --local-dir .\n"
            "puis passer --test splits/test_real.csv") from None
    return ({cle_arabe(x["darija"]) for x in lignes},
            {cle_latine(x["transliteration"]) for x in lignes})


def exporter(registre, sortie, test=TEST, avec_confirmations=False):
    """Ecrit le fichier d'entrainement et rend le bilan des lignes."""
    cles_ar, cles_la = cles_du_test(test)
    lignes = registre.a_exporter(avec_confirmations)
    gardees = [x for x in lignes
               if cle_arabe(x["darija"]) not in cles_ar
               and cle_latine(x["entree_propre"]) not in cles_la]
    with open(sortie, "w", encoding="utf-8-sig", newline="") as f:
        ecrivain = csv.DictWriter(f, fieldnames=COLONNES)
        ecrivain.writeheader()
        for x in gardees:
            ecrivain.writerow({
                "id": f"registre_{x['id']:06d}",
                "transliteration": x["entree_propre"],
                "darija_norm": normaliser_cible(x["darija"]),
                "darija": x["darija"],
                "source": "registre",
                "sortie_modele": x["sortie"],
                "confiance": x["confiance"],
                "revision": x["revision"],
            })
    return {"corrigees": sum(x["statut"] == "corrigee" for x in lignes),
            "confirmees": sum(x["statut"] == "confirmee" for x in lignes),
            "retirees_test": len(lignes) - len(gardees),
            "ecrites": len(gardees)}


def main(arguments=None):
    p = argparse.ArgumentParser(
        description="Exporte le registre de validation pour le réentraînement.")
    p.add_argument("sortie", help="fichier CSV à écrire")
    p.add_argument("--avec-confirmations", action="store_true",
                   help="ajouter les sorties confirmées, comme exemples justes")
    p.add_argument("--test", default=TEST,
                   help="jeu de test à épargner (défaut : test_real.csv)")
    a = p.parse_args(arguments)
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL non définie : quelle base exporter ?")
    bilan = exporter(Registre(url), a.sortie, a.test, a.avec_confirmations)
    print(a.sortie)
    print(f"  corrigées             {bilan['corrigees']:>6}")
    print(f"  confirmées            {bilan['confirmees']:>6}")
    print(f"  retirées, jeu de test {bilan['retirees_test']:>6}")
    print(f"  écrites               {bilan['ecrites']:>6}")
    return bilan


if __name__ == "__main__":
    main()
```

- [ ] **Étape 4 : vérifier**

Lancer : `py -m pytest tests/test_exporter.py -q`
Attendu : 6 réussis.

- [ ] **Étape 5 : commit**

```bash
git add registre/exporter.py tests/test_exporter.py
git commit -m "Export du registre pour le reentrainement, jeu de test epargne"
```

---

### Tâche 4 : le service enregistre chaque prédiction

**Fichiers :**
- Modifier : `service/config.py` (fin de fichier), `service/api.py`
- Test : `tests/test_service.py` (ajouts en fin de fichier)

**Interfaces :**
- Consomme : `Registre(url)`, `Registre.creer_tables()`,
  `Registre.enregistrer(lignes) -> list[int]`.
- Produit : `creer_app(normaliseur=None, calibration=None, registre=None)`.
  `Resultat.id` vaut `int | None`, et `/sante` porte une clé booléenne
  `registre`.

- [ ] **Étape 1 : écrire les tests qui échouent**

Ajouter en tête de `tests/test_service.py`, après les imports existants :

```python
from sqlalchemy.exc import OperationalError

import service.api
from registre.depot import Registre
```

et en fin de fichier :

```python
# --- Registre de validation humaine -------------------------------------------

def url_sqlite(tmp_path):
    return f"sqlite:///{(tmp_path / 'registre.db').as_posix()}"


@pytest.fixture
def registre(tmp_path):
    return Registre(url_sqlite(tmp_path))


@pytest.fixture
def client_registre(faux, registre):
    with TestClient(creer_app(faux, CALIBRATION, registre)) as c:
        yield c


def test_une_prediction_enregistree(client_registre, registre):
    d = client_registre.post(
        "/normaliser", json={"texte": "  chrit 3   tomobilat "}).json()
    p = registre.prediction(d["id"])
    assert p["entree"] == "  chrit 3   tomobilat "
    assert p["entree_propre"] == "chrit 3 tomobilat"
    assert p["sortie"] == d["sortie"] == "شريت 3 طوموبيلات"
    assert p["confiance"] == d["confiance"]
    assert p["confiance_brute"] == d["confiance_brute"]
    assert p["complete"] is True and p["revision"] == "rev-test"
    assert p["statut"] == "a_valider"


def test_un_lot_enregistre_n_predictions(client_registre, registre):
    textes = ["kanbghik bzaf", "chrit 3 tomobilat", "wach"]
    resultats = client_registre.post(
        "/normaliser/lot", json={"textes": textes}).json()["resultats"]
    ids = [x["id"] for x in resultats]
    assert len(set(ids)) == 3
    assert [registre.prediction(i)["entree"] for i in ids] == textes
    assert [registre.prediction(i)["sortie"] for i in ids] == [
        x["sortie"] for x in resultats]
    assert registre.a_valider()[1] == 3


class RegistreEnPanne(Registre):
    def enregistrer(self, lignes):
        raise OperationalError("INSERT INTO predictions", {},
                               ConnectionRefusedError("base arrêtée"))


def test_base_indisponible_503(faux, tmp_path):
    with TestClient(creer_app(faux, CALIBRATION,
                              RegistreEnPanne(url_sqlite(tmp_path)))) as c:
        r = c.post("/normaliser", json={"texte": "kanbghik bzaf"})
        assert r.status_code == 503 and "registre" in r.json()["detail"]
        lot = c.post("/normaliser/lot", json={"textes": ["wach"]})
        assert lot.status_code == 503


def test_sans_database_url_rien_n_est_enregistre(faux, monkeypatch):
    monkeypatch.setattr(service.api, "DATABASE_URL", None)
    with TestClient(creer_app(faux, CALIBRATION)) as c:
        assert c.post("/normaliser", json={"texte": "wach"}).json()["id"] is None
        assert c.get("/sante").json()["registre"] is False


def test_database_url_active_le_registre(faux, tmp_path, monkeypatch):
    monkeypatch.setattr(service.api, "DATABASE_URL", url_sqlite(tmp_path))
    with TestClient(creer_app(faux, CALIBRATION)) as c:
        d = c.post("/normaliser", json={"texte": "kanbghik bzaf"}).json()
        assert c.get("/sante").json()["registre"] is True
    assert Registre(url_sqlite(tmp_path)).prediction(d["id"])["sortie"] == d["sortie"]


def test_503_documente_dans_openapi(client):
    chemins = client.get("/openapi.json").json()["paths"]
    for route in ("/normaliser", "/normaliser/lot"):
        assert "503" in chemins[route]["post"]["responses"]
```

- [ ] **Étape 2 : vérifier qu'ils échouent**

Lancer : `py -m pytest tests/test_service.py -q`
Attendu : ÉCHEC des 6 nouveaux tests. `creer_app` refuse un troisième argument,
`service.api` n'a pas de `DATABASE_URL`, et les tests existants passent
toujours.

- [ ] **Étape 3 : ajouter le réglage**

En fin de `service/config.py` :

```python

# Base du registre de validation humaine. Definie, chaque prediction y est
# enregistree avant que la reponse parte ; absente, le service n'enregistre
# rien, comme dans les tests ou hors de la chaine docker compose.
DATABASE_URL = os.environ.get("DATABASE_URL")
```

- [ ] **Étape 4 : enregistrer dans `service/api.py`**

Imports, à la place des lignes 8 à 15 :

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import InterfaceError, OperationalError

from registre.depot import Registre
from service.calibration import Calibration
from service.config import CALIBRATION, DATABASE_URL, MAX_LOT
from service.normaliseur import MAX_OCTETS, Normaliseur, nettoyer
```

En fin de `DESCRIPTION`, avant les `"""` fermants, ajouter le paragraphe :

```
Relié au registre de validation, le service enregistre chaque prédiction
avant de répondre, et la réponse porte son `id` : elle attend d'être
confirmée ou corrigée sur l'interface de validation. Si le registre ne répond
pas, le service renvoie 503 plutôt qu'une prédiction qui échapperait à la
validation.
```

Dans `Resultat`, premier champ :

```python
    id: int | None = Field(
        default=None,
        description="Identifiant de la prédiction dans le registre de "
                    "validation ; null si le service n'enregistre pas.")
```

Après la classe `ReponseLot`, ajouter :

```python
INDISPONIBLE = {503: {"description": "Registre de validation injoignable : "
                                     "aucune prédiction n'est rendue."}}
```

Remplacer `creer_app` en entier par :

```python
def creer_app(normaliseur=None, calibration=None, registre=None):
    """Construit l'application. Les tests y injectent un faux modele et un
    registre SQLite ; sans registre injecte, DATABASE_URL decide."""

    @asynccontextmanager
    async def cycle(app):
        n = normaliseur or Normaliseur()
        c = calibration or Calibration.charger(CALIBRATION)
        # Un rang calcule sur les confiances d'un autre modele serait faux sans
        # que rien ne le signale : on refuse de demarrer.
        if (c.modele, c.revision) != (n.nom, n.revision):
            raise RuntimeError(
                f"calibration de {c.modele}@{c.revision}, modèle servi "
                f"{n.nom}@{n.revision} : relancer python -m service.calibrer")
        r = registre
        if r is None and DATABASE_URL:
            r = Registre(DATABASE_URL)
        if r is not None:
            r.creer_tables()
        app.state.normaliseur, app.state.calibration = n, c
        app.state.registre = r
        yield

    app = FastAPI(title="ByT5 Arabizi Normalizer", version="1.0.0",
                  description=DESCRIPTION, lifespan=cycle)

    def traiter(request, textes):
        n, c = request.app.state.normaliseur, request.app.state.calibration
        r = request.app.state.registre
        propres = [nettoyer(t) for t in textes]
        sorties = n.normaliser(propres)
        lignes = [{"entree": t, "entree_propre": p, "sortie": s,
                   "confiance": c.rang(brute),
                   "confiance_brute": round(brute, 4), "complete": complete,
                   "revision": n.revision}
                  for t, p, (s, brute, complete) in zip(textes, propres, sorties)]
        ids = [None] * len(lignes)
        if r is not None:
            # La directive veut que toute prediction passe par la validation :
            # une prediction rendue sans etre enregistree y echapperait sans
            # que rien ne le signale.
            try:
                ids = r.enregistrer(lignes)
            except (OperationalError, InterfaceError) as erreur:
                raise HTTPException(
                    503, "registre de validation injoignable : la prédiction "
                         "n'est pas rendue faute de pouvoir être enregistrée"
                ) from erreur
        return [Resultat(id=i, entree=x["entree"], sortie=x["sortie"],
                         confiance=x["confiance"],
                         confiance_brute=x["confiance_brute"],
                         complete=x["complete"])
                for i, x in zip(ids, lignes)]

    @app.post("/normaliser", response_model=Resultat,
              summary="Normaliser une phrase", responses=INDISPONIBLE)
    def normaliser(requete: Requete, request: Request):
        return traiter(request, [requete.texte])[0]

    @app.post("/normaliser/lot", response_model=ReponseLot,
              summary="Normaliser un lot de phrases (EF-06)",
              responses=INDISPONIBLE)
    def normaliser_lot(requete: RequeteLot, request: Request):
        return ReponseLot(resultats=traiter(request, requete.textes))

    @app.get("/sante", summary="État du service et modèle servi")
    def sante(request: Request):
        n, c = request.app.state.normaliseur, request.app.state.calibration
        return {"statut": "pret", "modele": n.nom, "revision": n.revision,
                "appareil": n.appareil, "phrases_calibration": len(c.scores),
                "registre": request.app.state.registre is not None}

    return app
```

- [ ] **Étape 5 : vérifier**

Lancer : `py -m pytest -q`
Attendu : tout réussit, soit 35 + 14 + 6 + 6 = 61.

- [ ] **Étape 6 : commit**

```bash
git add service/config.py service/api.py tests/test_service.py
git commit -m "Le service enregistre chaque prediction, et renvoie 503 sans registre"
```

---

### Tâche 5 : l'interface de validation

**Fichiers :**
- Créer : `validation/__init__.py`, `validation/app.py`,
  `validation/gabarits/base.html`, `validation/gabarits/pages.html`,
  `validation/gabarits/liste.html`, `validation/gabarits/prediction.html`,
  `validation/gabarits/historique.html`
- Test : `tests/test_validation.py`

**Interfaces :**
- Consomme : `Registre` (`a_valider`, `premiere_a_valider`, `prediction`,
  `valider`, `historique`), `DECISIONS`, `MOTIFS`, `contient_vocalisation`.
- Produit : `creer_app(registre=None)` et `app`, dans `validation.app`.

Routes : `GET /` (liste), `GET /suivante`, `GET /prediction/{id}`,
`POST /prediction/{id}` (champs `decision`, `texte`, `motif`, `validateur`,
`remarque`, `depuis`), `GET /historique`.

- [ ] **Étape 1 : écrire les tests qui échouent**

`tests/test_validation.py` :

```python
"""Tests de l'interface de validation, sur SQLite, sans modele."""

import re

import pytest
from fastapi.testclient import TestClient

from registre.depot import Registre
from validation.app import creer_app

SORTIE = "شريت 3 طوموبيلات"


def ligne(entree, sortie, confiance):
    return {"entree": entree, "entree_propre": entree, "sortie": sortie,
            "confiance": confiance, "confiance_brute": 0.95, "complete": True,
            "revision": "rev-test"}


@pytest.fixture
def registre(tmp_path):
    r = Registre(f"sqlite:///{(tmp_path / 'registre.db').as_posix()}")
    r.creer_tables()
    return r


@pytest.fixture
def client(registre):
    with TestClient(creer_app(registre)) as c:
        yield c


@pytest.fixture
def ids(registre):
    return registre.enregistrer([
        ligne("chrit 3 tomobilat", SORTIE, 0.80),
        ligne("kanbghik bzaf", "كنبغيك بزاف", 0.10),
        ligne("wach nta mzyan", "واش نتا مزيان", 0.50)])


def decider(client, i, **champs):
    donnees = {"validateur": "Awa", "texte": "", "motif": "", **champs}
    return client.post(f"/prediction/{i}", data=donnees, follow_redirects=False)


def erreur(page):
    m = re.search(r'<p class="erreur" role="alert">(.*?)</p>', page, re.S)
    return m.group(1) if m else None


def test_liste_triee_par_confiance_croissante(client, ids):
    page = client.get("/").text
    positions = [page.index(x) for x in
                 ("kanbghik bzaf", "wach nta mzyan", "chrit 3 tomobilat")]
    assert positions == sorted(positions)
    assert "3 prédictions à valider" in page


def test_liste_par_pages_de_vingt(client, registre):
    registre.enregistrer([ligne(f"jomla {i}", "جملة", i / 100) for i in range(25)])
    assert client.get("/").text.count('class="prediction"') == 20
    assert client.get("/?page=2").text.count('class="prediction"') == 5


def test_suivante_ouvre_la_moins_sure(client, ids):
    r = client.get("/suivante", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/prediction/{ids[1]}"


def test_suivante_sans_rien_a_valider(client):
    assert client.get("/suivante", follow_redirects=False).headers["location"] == "/"


def test_page_pre_remplie_de_droite_a_gauche(client, ids):
    page = client.get(f"/prediction/{ids[0]}").text
    assert re.search(rf'<textarea[^>]*dir="rtl"[^>]*>{SORTIE}</textarea>', page)
    for lettre in "ڭڤ؟،":
        assert f'data-lettre="{lettre}"' in page


def test_confirmer(client, registre, ids):
    r = decider(client, ids[0], decision="confirmee", texte=SORTIE)
    assert r.status_code == 303 and r.headers["location"] == "/suivante"
    assert registre.prediction(ids[0])["statut"] == "confirmee"


def test_correction_visible_apres_rechargement(client, registre, ids):
    decider(client, ids[0], decision="corrigee", texte="شريت  3 طوموبيلات ديال\r\n")
    p = registre.prediction(ids[0])
    assert p["statut"] == "corrigee"
    assert p["derniere"]["correction"] == "شريت 3 طوموبيلات ديال"
    assert ">شريت 3 طوموبيلات ديال</textarea>" in client.get(f"/prediction/{ids[0]}").text


def test_ecarter_avec_motif(client, registre, ids):
    decider(client, ids[2], decision="ecartee", motif="donnee_personnelle")
    p = registre.prediction(ids[2])
    assert p["statut"] == "ecartee" and p["derniere"]["motif"] == "donnee_personnelle"


@pytest.mark.parametrize("champs, message", [
    ({"decision": "ecartee"}, "motif"),
    ({"decision": "confirmee", "texte": "شي حاجة اخرى"}, "Corriger"),
    ({"decision": "corrigee", "texte": SORTIE}, "Confirmer"),
    ({"decision": "corrigee", "texte": "   "}, "Écarter"),
    ({"decision": "corrigee", "texte": "شْريت 3 طوموبيلات"}, "vocalisation"),
    ({"decision": "confirmee", "texte": SORTIE, "validateur": " "}, "nom"),
    ({"decision": "validee", "texte": SORTIE}, "inconnue"),
])
def test_decisions_refusees(client, registre, ids, champs, message):
    r = decider(client, ids[0], **champs)
    assert r.status_code == 422 and message in erreur(r.text)
    assert registre.prediction(ids[0])["statut"] == "a_valider"
    assert registre.historique()[1] == 0


def test_saisie_conservee_apres_refus(client, ids):
    page = decider(client, ids[0], decision="corrigee", texte="شْريت").text
    assert ">شْريت</textarea>" in page


def test_nom_garde_dans_un_cookie(client, ids):
    decider(client, ids[0], decision="confirmee", texte=SORTIE, validateur="سعاد")
    assert 'value="سعاد"' in client.get(f"/prediction/{ids[1]}").text


def test_historique_garde_chaque_decision(client, registre, ids):
    decider(client, ids[0], decision="corrigee", texte="شريت 3 طوموبيلات ديال")
    decider(client, ids[0], decision="confirmee", texte=SORTIE, depuis="historique")
    page = client.get("/historique").text
    assert page.count('class="decision"') == 2
    assert f"/prediction/{ids[0]}?depuis=historique" in page
    assert registre.prediction(ids[0])["statut"] == "confirmee"


def test_retour_vers_l_historique(client, ids):
    r = decider(client, ids[0], decision="confirmee", texte=SORTIE, depuis="historique")
    assert r.headers["location"] == "/historique"


def test_retour_inconnu_ignore(client, ids):
    r = decider(client, ids[0], decision="confirmee", texte=SORTIE,
                depuis="https://exemple.org")
    assert r.headers["location"] == "/suivante"


def test_prediction_inconnue(client):
    assert client.get("/prediction/999").status_code == 404
    assert decider(client, 999, decision="confirmee").status_code == 404


def test_texte_echappe(client, registre):
    [i] = registre.enregistrer([ligne("<script>alert(1)</script>", "x", 0.1)])
    page = client.get(f"/prediction/{i}").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_sans_database_url_refuse_de_demarrer(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        with TestClient(creer_app()):
            pass
```

- [ ] **Étape 2 : vérifier qu'ils échouent**

Lancer : `py -m pytest tests/test_validation.py -q`
Attendu : ÉCHEC, `ModuleNotFoundError: No module named 'validation'`.

- [ ] **Étape 3 : écrire l'application**

`validation/__init__.py` :

```python
"""Interface de validation humaine des predictions du service."""
```

`validation/app.py` :

```python
"""Interface de validation humaine des predictions du service.

Lancer depuis la racine du depot, la base designee par DATABASE_URL :
    uvicorn validation.app:app --port 8001

Toute prediction enregistree par le service attend ici qu'un validateur la
confirme, la corrige ou l'ecarte. Les moins sures sont presentees d'abord.
Pas d'authentification : l'outil tourne en local, et docker compose ne
publie son port que sur la machine.
"""

import os
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from donnees.normalisation import contient_vocalisation
from registre.depot import Registre
from registre.schema import DECISIONS, MOTIFS

PAR_PAGE = 20
COOKIE = "validateur"
UN_AN = 365 * 24 * 3600
# Pages ou revenir apres une decision. Liste fermee : une adresse reprise telle
# quelle du formulaire ferait de l'interface un redirecteur ouvert.
RETOURS = {"suivante": "/suivante", "historique": "/historique"}
GABARITS = Jinja2Templates(directory=Path(__file__).parent / "gabarits")


def propre(texte):
    """Le texte tel qu'il est compare et enregistre : NFKC, espaces reduites."""
    return " ".join(unicodedata.normalize("NFKC", texte or "").split())


def verifier(decision, texte, sortie, motif, validateur):
    """Rend (correction, None) si la decision tient, (None, message) sinon.

    Rien n'est corrige a la place du validateur : une saisie qui ne correspond
    pas au bouton choisi est refusee avec un message, jamais reinterpretee.
    """
    if not propre(validateur):
        return None, "Indiquer votre nom avant de valider."
    if decision not in DECISIONS:
        return None, "Décision inconnue."
    texte, sortie = propre(texte), propre(sortie)
    if decision == "confirmee":
        if texte != sortie:
            return None, ("Le texte a été modifié : utiliser « Corriger », ou "
                          "rétablir la sortie du modèle.")
        return None, None
    if decision == "ecartee":
        if motif not in MOTIFS:
            return None, "Choisir un motif pour écarter la phrase."
        return None, None
    if not texte:
        return None, ("La correction est vide : utiliser « Écarter » si la "
                      "phrase est inutilisable.")
    if texte == sortie:
        return None, "Le texte n'a pas été modifié : utiliser « Confirmer »."
    if contient_vocalisation(texte):
        return None, ("La correction contient de la vocalisation, que la "
                      "convention de sortie n'écrit pas. Retirer les signes : "
                      "rien n'est retiré à votre place.")
    return texte, None


def nombre_de_pages(total):
    return max(1, -(-total // PAR_PAGE))


def creer_app(registre=None):
    """Construit l'application. Les tests y injectent un registre SQLite."""

    @asynccontextmanager
    async def cycle(app):
        r = registre
        if r is None:
            url = os.environ.get("DATABASE_URL")
            if not url:
                raise RuntimeError("DATABASE_URL non définie : l'interface "
                                   "n'a aucun registre à lire.")
            r = Registre(url)
        r.creer_tables()
        app.state.registre = r
        yield

    app = FastAPI(title="Validation des prédictions", version="1.0.0",
                  lifespan=cycle)

    def lire(request, prediction_id):
        p = request.app.state.registre.prediction(prediction_id)
        if p is None:
            raise HTTPException(404, "prédiction inconnue")
        return p

    def page_prediction(request, p, texte, validateur, depuis, message=None):
        return GABARITS.TemplateResponse(
            request, "prediction.html",
            {"p": p, "texte": texte, "validateur": validateur,
             "depuis": depuis, "erreur": message, "motifs": MOTIFS},
            status_code=422 if message else 200)

    @app.get("/", summary="Prédictions à valider, les moins sûres d'abord")
    def liste(request: Request, page: int = Query(1, ge=1)):
        lignes, total = request.app.state.registre.a_valider(page, PAR_PAGE)
        return GABARITS.TemplateResponse(request, "liste.html", {
            "lignes": lignes, "total": total, "page": page,
            "pages": nombre_de_pages(total)})

    @app.get("/suivante", summary="Ouvrir la prédiction la moins sûre en attente")
    def suivante(request: Request):
        i = request.app.state.registre.premiere_a_valider()
        return RedirectResponse(f"/prediction/{i}" if i else "/", status_code=303)

    @app.get("/prediction/{prediction_id}", summary="Ouvrir une prédiction")
    def ouvrir(request: Request, prediction_id: int, depuis: str = "suivante"):
        p = lire(request, prediction_id)
        d = p["derniere"]
        texte = d["correction"] if d and d["decision"] == "corrigee" else p["sortie"]
        return page_prediction(request, p, texte,
                               unquote(request.cookies.get(COOKIE, "")),
                               depuis if depuis in RETOURS else "suivante")

    @app.post("/prediction/{prediction_id}",
              summary="Confirmer, corriger ou écarter une prédiction")
    def decider(request: Request, prediction_id: int,
                decision: str = Form(...), texte: str = Form(""),
                motif: str = Form(""), validateur: str = Form(""),
                remarque: str = Form(""), depuis: str = Form("suivante")):
        p = lire(request, prediction_id)
        depuis = depuis if depuis in RETOURS else "suivante"
        correction, message = verifier(decision, texte, p["sortie"], motif,
                                       validateur)
        if message:
            return page_prediction(request, p, texte, validateur, depuis, message)
        nom = propre(validateur)
        request.app.state.registre.valider(
            prediction_id, decision, nom, correction=correction,
            motif=motif or None, remarque=remarque)
        reponse = RedirectResponse(RETOURS[depuis], status_code=303)
        reponse.set_cookie(COOKIE, quote(nom), max_age=UN_AN, httponly=True,
                           samesite="lax")
        return reponse

    @app.get("/historique", summary="Décisions prises, les plus récentes d'abord")
    def historique(request: Request, page: int = Query(1, ge=1)):
        lignes, total = request.app.state.registre.historique(page, PAR_PAGE)
        return GABARITS.TemplateResponse(request, "historique.html", {
            "lignes": lignes, "total": total, "page": page,
            "pages": nombre_de_pages(total), "motifs": MOTIFS})

    return app


app = creer_app()
```

- [ ] **Étape 4 : écrire les gabarits**

Starlette active l'échappement automatique pour les fichiers `.html`.
`test_texte_echappe` le vérifie.

`validation/gabarits/base.html` :

```html
<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block titre %}Validation{% endblock %} · ByT5 Arabizi</title>
<style>
  :root { --encre: #1d2330; --doux: #5d6675; --trait: #d9dde3;
          --fond: #f7f8fa; --carte: #ffffff; --accent: #1f5fbf; --alerte: #b3261e; }
  body { margin: 0; font: 16px/1.5 system-ui, sans-serif; color: var(--encre);
         background: var(--fond); }
  header, main { max-width: 960px; margin: 0 auto; padding: 16px; }
  nav a { margin-right: 16px; color: var(--accent); }
  table { width: 100%; border-collapse: collapse; background: var(--carte); }
  th, td { padding: 8px; border-bottom: 1px solid var(--trait); text-align: left;
           vertical-align: top; }
  .ar { direction: rtl; text-align: right; font-size: 1.15em; }
  textarea.ar { width: 100%; min-height: 5em; box-sizing: border-box;
                font: inherit; font-size: 1.3em; }
  .erreur { color: var(--alerte); border: 1px solid var(--alerte); padding: 8px;
            background: var(--carte); }
  .actions button, .lettres button { margin: 4px 8px 4px 0; padding: 8px 14px; }
  .doux { color: var(--doux); }
</style>
</head>
<body>
<header><nav>
  <a href="/">À valider</a><a href="/suivante">Valider la suivante</a><a href="/historique">Historique</a>
</nav></header>
<main>{% block contenu %}{% endblock %}</main>
</body>
</html>
```

`validation/gabarits/pages.html` :

```html
{% if pages > 1 %}
<p>
  {% if page > 1 %}<a href="?page={{ page - 1 }}">← précédentes</a>{% endif %}
  page {{ page }} sur {{ pages }}
  {% if page < pages %}<a href="?page={{ page + 1 }}">suivantes →</a>{% endif %}
</p>
{% endif %}
```

`validation/gabarits/liste.html` :

```html
{% extends "base.html" %}
{% block titre %}À valider{% endblock %}
{% block contenu %}
<h1>{{ total }} prédiction{{ "s" if total > 1 }} à valider</h1>
<p class="doux">Les moins sûres d'abord. La confiance est un rang : à 0,10, neuf
phrases sur dix du jeu de développement étaient plus sûres.</p>
{% if lignes %}
<table>
<thead><tr><th>Confiance</th><th>Arabizi</th><th>Sortie du modèle</th><th></th></tr></thead>
<tbody>
{% for l in lignes %}
<tr class="prediction">
  <td>{{ "%.2f"|format(l.confiance) }}</td>
  <td>{{ l.entree }}</td>
  <td class="ar" dir="rtl" lang="ar">{{ l.sortie }}</td>
  <td><a href="/prediction/{{ l.id }}">Ouvrir</a></td>
</tr>
{% endfor %}
</tbody>
</table>
{% include "pages.html" %}
{% else %}
<p>Rien à valider.</p>
{% endif %}
{% endblock %}
```

`validation/gabarits/prediction.html` :

```html
{% extends "base.html" %}
{% block titre %}Prédiction {{ p.id }}{% endblock %}
{% block contenu %}
<h1>Prédiction {{ p.id }}</h1>
<p class="doux">Confiance {{ "%.2f"|format(p.confiance) }} · statut : {{ p.statut }}
{%- if not p.complete %} · <strong>sortie tronquée</strong>{% endif %}</p>
{% if p.derniere %}
<p class="doux">Dernière décision : {{ p.derniere.decision }} par {{ p.derniere.validateur }}
{%- if p.derniere.motif %} ({{ motifs[p.derniere.motif] }}){% endif %}.</p>
{% endif %}
{% if erreur %}<p class="erreur" role="alert">{{ erreur }}</p>{% endif %}
<p><strong>Arabizi</strong> : {{ p.entree }}</p>
<p><strong>Sortie du modèle</strong> :
  <span id="sortie" class="ar" dir="rtl" lang="ar">{{ p.sortie }}</span></p>
<form method="post">
  <input type="hidden" name="depuis" value="{{ depuis }}">
  <p><label>Votre nom <input name="validateur" value="{{ validateur }}" required></label></p>
  <p><label for="texte">Texte retenu</label></p>
  <textarea id="texte" name="texte" class="ar" dir="rtl" lang="ar">{{ texte }}</textarea>
  <p class="lettres">
    {% for lettre in ["ڭ", "ڤ", "؟", "،"] %}<button type="button" class="lettre" data-lettre="{{ lettre }}">{{ lettre }}</button>{% endfor %}
    <button type="button" id="retablir">Rétablir la sortie du modèle</button>
  </p>
  <p><label>Motif, pour écarter
    <select name="motif">
      <option value="">—</option>
      {% for code, libelle in motifs.items() %}<option value="{{ code }}">{{ libelle }}</option>{% endfor %}
    </select></label></p>
  <p><label>Remarque <input name="remarque"></label></p>
  <p class="actions">
    <button name="decision" value="confirmee">Confirmer</button>
    <button name="decision" value="corrigee">Corriger</button>
    <button name="decision" value="ecartee">Écarter</button>
  </p>
</form>
<script>
  const zone = document.getElementById("texte");
  document.querySelectorAll(".lettre").forEach(b => b.addEventListener("click", () => {
    zone.setRangeText(b.dataset.lettre, zone.selectionStart, zone.selectionEnd, "end");
    zone.focus();
  }));
  document.getElementById("retablir").addEventListener("click", () => {
    zone.value = document.getElementById("sortie").textContent;
    zone.focus();
  });
</script>
{% endblock %}
```

`validation/gabarits/historique.html` :

```html
{% extends "base.html" %}
{% block titre %}Historique{% endblock %}
{% block contenu %}
<h1>Historique : {{ total }} décision{{ "s" if total > 1 }}</h1>
<p class="doux">Chaque décision est gardée. Pour une phrase revue plusieurs fois,
la plus récente fait foi.</p>
{% if lignes %}
<table>
<thead><tr><th>Date</th><th>Validateur</th><th>Décision</th><th>Arabizi</th><th>Texte retenu</th><th></th></tr></thead>
<tbody>
{% for l in lignes %}
<tr class="decision">
  <td>{{ l.cree_le.strftime("%Y-%m-%d %H:%M") if l.cree_le }}</td>
  <td>{{ l.validateur }}</td>
  <td>{{ l.decision }}{% if l.motif %} ({{ motifs[l.motif] }}){% endif %}</td>
  <td>{{ l.entree }}</td>
  <td class="ar" dir="rtl" lang="ar">{% if l.decision == "corrigee" %}{{ l.correction }}{% elif l.decision == "confirmee" %}{{ l.sortie }}{% endif %}</td>
  <td><a href="/prediction/{{ l.prediction_id }}?depuis=historique">Rouvrir</a></td>
</tr>
{% endfor %}
</tbody>
</table>
{% include "pages.html" %}
{% else %}
<p>Aucune décision pour l'instant.</p>
{% endif %}
{% endblock %}
```

- [ ] **Étape 5 : vérifier**

Lancer : `py -m pytest -q`
Attendu : tout réussit, soit 61 + 23 = 84.

- [ ] **Étape 6 : regarder les pages**

Lancer :

```bash
py -c "from registre.depot import Registre; r = Registre('sqlite:///essai.db'); r.creer_tables(); r.enregistrer([{'entree': 'kanbghik bzaf', 'entree_propre': 'kanbghik bzaf', 'sortie': 'كنبغيك بزاف', 'confiance': 0.12, 'confiance_brute': 0.95, 'complete': True, 'revision': 'essai'}])"
DATABASE_URL=sqlite:///essai.db py -m uvicorn validation.app:app --port 8001
```

Ouvrir <http://localhost:8001>, puis corriger la phrase, la rouvrir depuis
l'historique et vérifier le champ de droite à gauche et les quatre boutons.
Arrêter le serveur, puis supprimer `essai.db`.

- [ ] **Étape 7 : commit**

```bash
git add validation/ tests/test_validation.py
git commit -m "Interface de validation : liste par confiance, decisions, historique"
```

---

### Tâche 6 : les conteneurs

**Fichiers :**
- Créer : `requirements-validation.txt`, `validation/Dockerfile`, `compose.yaml`
- Modifier : `requirements-service.txt`, `Dockerfile`, `.dockerignore`

**Interfaces :**
- Consomme : `service.api:app` (port 8000), `validation.app:app` (port 8001),
  `python -m registre.exporter`, et la variable `DATABASE_URL` des deux
  conteneurs applicatifs.

- [ ] **Étape 1 : dépendances**

En fin de `requirements-service.txt` :

```
# Registre de validation humaine : le service y enregistre chaque prediction
# quand DATABASE_URL est definie.
sqlalchemy==2.0.51
psycopg[binary]==3.3.9
```

`requirements-validation.txt` :

```
# Dependances de l'interface de validation, versions figees (ENF-04).
# Ni torch ni transformers : elle lit et ecrit le registre, sans modele.
fastapi==0.136.1
uvicorn==0.44.0
pydantic==2.13.0
jinja2==3.1.6
python-multipart==0.0.26
sqlalchemy==2.0.51
psycopg[binary]==3.3.9
```

- [ ] **Étape 2 : l'image du service embarque le registre**

Dans `Dockerfile`, après `COPY service/ service/` :

```dockerfile
COPY registre/__init__.py registre/schema.py registre/depot.py registre/
```

Dans `.dockerignore`, après `!service/` :

```
!requirements-validation.txt
!registre/
!validation/
!donnees/__init__.py
!donnees/normalisation.py
```

- [ ] **Étape 3 : l'image de validation**

`validation/Dockerfile` :

```dockerfile
# syntax=docker/dockerfile:1
# Image de l'interface de validation : ni torch ni modele. Construite par
# docker compose, depuis la racine du depot (compose.yaml).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY requirements-validation.txt .
RUN pip install -r requirements-validation.txt
RUN useradd --create-home validation

COPY donnees/__init__.py donnees/normalisation.py donnees/
COPY registre/ registre/
COPY validation/__init__.py validation/app.py validation/
COPY validation/gabarits/ validation/gabarits/

USER validation
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/', timeout=4)"

CMD ["uvicorn", "validation.app:app", "--host", "0.0.0.0", "--port", "8001"]
```

`PYTHONPATH=/app` permet de lancer l'export depuis un autre répertoire de
travail (étape 6).

- [ ] **Étape 4 : la composition**

`compose.yaml` :

```yaml
# Chaine de validation humaine : le service, sa base, l'interface.
#   HF_TOKEN=... docker compose up --build -d
# Le jeton ne sert qu'a construire l'image du service, le modele etant prive.
# Service : http://localhost:8000/docs    Validation : http://localhost:8001
# Les ports ne sont publies que sur la machine : l'interface n'a pas
# d'authentification. "docker compose down -v" efface le registre.

name: byt5-arabizi

services:
  base:
    image: postgres:17.12-alpine
    environment:
      POSTGRES_USER: registre
      POSTGRES_PASSWORD: ${REGISTRE_MOT_DE_PASSE:-registre}
      POSTGRES_DB: registre
    volumes:
      - registre:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U registre -d registre"]
      interval: 5s
      timeout: 3s
      retries: 12

  api:
    build:
      context: .
      secrets: [hf_token]
    environment:
      DATABASE_URL: postgresql+psycopg://registre:${REGISTRE_MOT_DE_PASSE:-registre}@base:5432/registre
    depends_on:
      base:
        condition: service_healthy
    ports:
      - "127.0.0.1:8000:8000"

  validation:
    build:
      context: .
      dockerfile: validation/Dockerfile
    environment:
      DATABASE_URL: postgresql+psycopg://registre:${REGISTRE_MOT_DE_PASSE:-registre}@base:5432/registre
    # L'api cree les tables a son demarrage : l'attendre evite que les deux
    # conteneurs les creent en meme temps.
    depends_on:
      api:
        condition: service_healthy
    ports:
      - "127.0.0.1:8001:8001"

volumes:
  registre:

secrets:
  hf_token:
    environment: HF_TOKEN
```

Lancer : `docker compose config --quiet`
Attendu : aucune sortie, code 0.

- [ ] **Étape 5 : démarrer et valider de bout en bout**

L'image du service se reconstruit en entier, modèle compris. Il faut
plusieurs gigaoctets de mémoire libre : arrêter d'abord les autres conteneurs.

```bash
export HF_TOKEN="$(py -c 'from huggingface_hub import get_token; print(get_token())')"
docker compose up --build -d
docker compose ps        # attendre trois conteneurs healthy, ou running pour base
curl -s -X POST localhost:8000/normaliser -H "Content-Type: application/json" \
     -d '{"texte": "kanbghik bzaf"}'
```

Attendu : un JSON avec `"id": 1`.

```bash
curl -s -o /dev/null -w "%{http_code} %{redirect_url}\n" -X POST localhost:8001/prediction/1 \
     --data-urlencode "decision=corrigee" --data-urlencode "texte=كنبغيك بزاف بزاف" \
     --data-urlencode "validateur=essai"
docker compose exec base psql -U registre -d registre -c "SELECT prediction_id, correction FROM registre_erreurs"
```

Attendu : `303 http://localhost:8001/suivante`, puis une ligne `1 | كنبغيك بزاف بزاف`.

- [ ] **Étape 6 : persistance, panne de la base, export**

```bash
docker compose down && docker compose up -d
curl -s localhost:8001/historique | grep -c 'class="decision"'
```

Attendu : `1` une fois les conteneurs revenus. Le registre a survécu au
redémarrage.

```bash
docker compose stop base
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/normaliser \
     -H "Content-Type: application/json" -d '{"texte": "wach"}'
docker compose start base
```

Attendu : `503` en moins de dix secondes. Après redémarrage de la base, la même
requête rend 200 : `pool_pre_ping` rouvre la connexion.

```bash
docker compose run --rm --no-deps -v "$(pwd -W):/travail" -w /travail validation \
    python -m registre.exporter registre.csv --test test_real.csv
docker compose run --rm --no-deps validation python -c "import torch"
```

Attendu : un bilan avec `écrites 1`, et `registre.csv` à la racine, ignoré par
git (`*.csv`). Le second appel échoue sur `ModuleNotFoundError`, ce qui
confirme que l'image de validation n'embarque pas torch.

- [ ] **Étape 7 : commit**

```bash
git add requirements-service.txt requirements-validation.txt Dockerfile .dockerignore validation/Dockerfile compose.yaml
git commit -m "docker compose : service, base Postgres et interface de validation"
```

---

### Tâche 7 : documentation et journal

**Fichiers :**
- Modifier : `README.md` (après la section `## Service`),
  `docs/project/validation-humaine.md` (ligne 3, puis fin de fichier),
  `docs/project/decisions.md` (nouvelle entrée en tête)

- [ ] **Étape 1 : README**

Avant `## Documentation`, insérer :

````markdown
## Validation humaine

Toute prédiction passe par une validation humaine, qui alimente un registre
d'erreurs réutilisable pour le réentraînement. La conception est décrite dans
`docs/project/validation-humaine.md`. Trois conteneurs : le service, une base
Postgres et l'interface de validation.

```bash
export HF_TOKEN=hf_...                     # pour construire l'image du service
docker compose up --build -d
```

| Adresse | Rôle |
|---|---|
| <http://localhost:8000/docs> | le service ; chaque réponse porte l'`id` de la prédiction enregistrée |
| <http://localhost:8001> | les prédictions à valider, les moins sûres d'abord |
| <http://localhost:8001/historique> | les décisions prises, chacune réouvrable |

Exporter les corrections pour l'entraînement (`--avec-confirmations` ajoute les
sorties confirmées). Les phrases du jeu de test sont retirées de l'export :

```bash
docker compose run --rm --no-deps -v "$PWD:/travail" -w /travail validation \
    python -m registre.exporter registre.csv --test test_real.csv
```

Le registre survit à `docker compose down`. Il est effacé par
`docker compose down -v`.
````

- [ ] **Étape 2 : la conception devient la référence**

Dans `docs/project/validation-humaine.md`, remplacer la ligne 3 par :

```markdown
Conception arrêtée le 16 novembre 2026, implémentée le 25 novembre : paquets
`registre/` et `validation/`, `compose.yaml`.
```

et ajouter en fin de fichier :

````markdown
## Écarts à la conception

- **Confirmer avec un texte modifié est refusé**, comme corriger sans rien
  modifier. Le bouton choisi et la saisie doivent concorder : deviner
  l'intention reviendrait à décider à la place du validateur.
- **Les ports ne sont publiés que sur `127.0.0.1`**, l'interface n'ayant pas
  d'authentification.
- **`validation` attend que `api` soit prêt**, pour que les tables ne soient
  pas créées par deux conteneurs à la fois.
````

- [ ] **Étape 3 : journal**

En tête des entrées de `docs/project/decisions.md`, juste après le premier
`---` :

```markdown
## 25 novembre 2026 — La validation humaine est en place

**Décision.** La directive *Score de confiance et boucle de feedback* est
implémentée telle que conçue le 16 novembre : chaque prédiction du service est
enregistrée dans une base Postgres dédiée avant que la réponse parte, une
interface la fait confirmer, corriger ou écarter, et
`python -m registre.exporter` rend les corrections au format d'entraînement.

**Ce qui a été tranché en cours de route.**

- Base injoignable : **503**, et non une prédiction rendue sans être
  enregistrée, qui échapperait à la validation sans que rien ne le signale.
- Postgres **17.12** et non 18 : la version 18 a déplacé son répertoire de
  données, et un volume monté à l'ancien chemin ne retiendrait rien.
- La normalisation des cibles et les clés de rapprochement passent dans
  `donnees/normalisation.py`, sans dépendance : l'image de validation n'a ni
  pandas ni torch. `make_splits.py` les réexporte, et rien ne change pour les
  autres scripts.
- L'export retire les phrases du jeu de test avec les clés de `make_splits.py`
  et affiche leur nombre. Une phrase de TerjamaBench soumise au service puis
  corrigée fausserait sinon l'évaluation.

**Vérifié.** 54 tests sur SQLite, sans modèle ni serveur. Avec
`docker compose`, une correction survit au redémarrage des conteneurs, l'arrêt
de la base fait rendre 503 au service, et l'export d'une correction écrit une
ligne.

---
```

Le nombre de tests de la section **Vérifié** est celui des tâches 1 à 5
(5 + 14 + 6 + 6 + 23 = 54) : le recompter après exécution, et corriger s'il
diffère.

- [ ] **Étape 4 : vérifier la documentation**

Lancer : `py -m sphinx -b html -q docs docs/_build/html` (Sphinx 9.2.0 est
installé ; si `sphinxcontrib.mermaid` manque, l'installer dans un venv du
scratchpad, jamais dans le Python de l'utilisateur).
Attendu : aucun avertissement nouveau. La page `project/validation-humaine` est
déjà dans le sommaire, et `plans/` est hors de `docs/`, donc hors de la
construction.

- [ ] **Étape 5 : commit et PR**

```bash
git add README.md docs/project/validation-humaine.md docs/project/decisions.md
git commit -m "Documentation et journal : la validation humaine est en place"
git push origin validation-humaine
gh pr create --base metriques-wer --title "Validation humaine et registre d'erreurs" --body-file - <<'FIN'
Implemente la directive *Score de confiance et boucle de feedback* du cadrage,
selon la conception de `docs/project/validation-humaine.md`.

- `registre/` : predictions et validations (append-only) dans une base dediee,
  vue `registre_erreurs`, export pour le reentrainement qui epargne le jeu de test.
- `service/` : chaque prediction est enregistree avant la reponse, qui porte
  son `id` ; 503 si la base ne repond pas ; rien ne change sans `DATABASE_URL`.
- `validation/` : pages de validation, les moins sures d'abord ; confirmer,
  corriger, ecarter avec motif ; historique.
- `compose.yaml` : api, base Postgres avec volume, validation, ports sur
  127.0.0.1 seulement.

Tests : `py -m pytest -q`, sur SQLite, sans modele. Verification `docker compose`
consignee dans le journal du 25 novembre.

A fusionner apres #23, sur laquelle elle est empilee.
FIN
```

La PR est ouverte sur `metriques-wer`, et non sur `main`, pour ne montrer que
ses propres changements. Elle se fusionne après la PR #23, et elle n'est pas
fusionnée ici.

---

## Hors périmètre

- **Effacement des données personnelles.** Une phrase écartée pour donnée
  personnelle reste en base. L'effacer ferait sortir une ligne de l'historique
  append-only : c'est à arbitrer, pas à décider ici.
- **Réentraînement automatique.** L'export produit le fichier ; le relancer
  reste un acte humain (`donnees/ajouter_source.py`, puis
  `modele/train_byt5.py`).
- **Diagrammes d'architecture C4** de l'ensemble du système : demandés par le
  cadrage, ils font l'objet d'un travail à part.
