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
