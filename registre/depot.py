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
        # Jointure explicite : validations figure deja parmi les colonnes, et
        # SQLAlchemy ne saurait plus de quel cote joindre.
        requete = (select(predictions, validations.c.correction)
                   .select_from(predictions.join(
                       validations, validations.c.id == derniere))
                   .where(predictions.c.statut.in_(statuts))
                   .order_by(predictions.c.id))
        with self.moteur.connect() as conn:
            lignes = conn.execute(requete).mappings().all()
        return [{**x, "darija": x["correction"] if x["statut"] == "corrigee"
                 else x["sortie"]} for x in lignes]
