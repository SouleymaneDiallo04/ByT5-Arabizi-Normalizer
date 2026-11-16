"""API REST du normaliseur (EF-04, EF-06), documentee au format OpenAPI.

Lancer depuis la racine du depot :
    uvicorn service.api:app --port 8000
La page Swagger est alors sur http://localhost:8000/docs.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

from service.calibration import Calibration
from service.config import CALIBRATION, MAX_LOT
from service.normaliseur import MAX_OCTETS, Normaliseur, nettoyer

DESCRIPTION = """
Transcrit l'**arabizi marocain**, le darija écrit en lettres latines et en
chiffres, en **darija en écriture arabe**. Le modèle change l'alphabet ; il ne
traduit pas.

La `confiance` est un **rang** : la part des phrases du jeu de développement
dont la sortie était moins sûre. À 0,10, neuf phrases sur dix étaient plus
sûres ; les rangs les plus bas désignent les sorties à faire relire.
"""


def verifier_texte(texte):
    """Refuse ce que le modele ne peut pas traiter en entier."""
    propre = nettoyer(texte)
    if not propre:
        raise ValueError("texte vide")
    octets = len(propre.encode("utf-8"))
    if octets > MAX_OCTETS:
        raise ValueError(
            f"{octets} octets, au-delà des {MAX_OCTETS} que lit le modèle : "
            "découper le texte en phrases")
    return texte


class Requete(BaseModel):
    texte: str = Field(description="Une phrase en arabizi.",
                       examples=["chrit 3 tomobilat"])

    @field_validator("texte")
    @classmethod
    def _verifier(cls, texte):
        return verifier_texte(texte)


class RequeteLot(BaseModel):
    textes: list[str] = Field(
        min_length=1, max_length=MAX_LOT,
        description=f"De 1 à {MAX_LOT} phrases en arabizi.",
        examples=[["chrit 3 tomobilat", "kanbghik bzaf"]])

    @field_validator("textes")
    @classmethod
    def _verifier(cls, textes):
        for i, texte in enumerate(textes):
            try:
                verifier_texte(texte)
            except ValueError as erreur:
                raise ValueError(f"phrase {i} : {erreur}") from None
        return textes


class Resultat(BaseModel):
    entree: str = Field(description="Le texte reçu, tel quel.")
    sortie: str = Field(description="Sa transcription en écriture arabe.")
    confiance: float = Field(
        ge=0, le=1,
        description="Rang de la confiance parmi les phrases du jeu de "
                    "développement, en centièmes (PO-05).")
    confiance_brute: float = Field(
        ge=0, le=1,
        description="Probabilité moyenne par octet produit. Elle se concentre "
                    "entre 0,92 et 1 et se lit par son rang.")
    complete: bool = Field(
        description="Faux si la sortie a atteint la longueur maximale avant que "
                    "le modèle termine la phrase : elle est alors tronquée.")


class ReponseLot(BaseModel):
    resultats: list[Resultat] = Field(
        description="Un résultat par phrase, dans l'ordre de la requête.")


def creer_app(normaliseur=None, calibration=None):
    """Construit l'application. Les tests y injectent un faux modele."""

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
        app.state.normaliseur, app.state.calibration = n, c
        yield

    app = FastAPI(title="ByT5 Arabizi Normalizer", version="1.0.0",
                  description=DESCRIPTION, lifespan=cycle)

    def traiter(request, textes):
        n, c = request.app.state.normaliseur, request.app.state.calibration
        sorties = n.normaliser([nettoyer(t) for t in textes])
        return [Resultat(entree=t, sortie=s, confiance=c.rang(brute),
                         confiance_brute=round(brute, 4), complete=complete)
                for t, (s, brute, complete) in zip(textes, sorties)]

    @app.post("/normaliser", response_model=Resultat,
              summary="Normaliser une phrase")
    def normaliser(requete: Requete, request: Request):
        return traiter(request, [requete.texte])[0]

    @app.post("/normaliser/lot", response_model=ReponseLot,
              summary="Normaliser un lot de phrases (EF-06)")
    def normaliser_lot(requete: RequeteLot, request: Request):
        return ReponseLot(resultats=traiter(request, requete.textes))

    @app.get("/sante", summary="État du service et modèle servi")
    def sante(request: Request):
        n, c = request.app.state.normaliseur, request.app.state.calibration
        return {"statut": "pret", "modele": n.nom, "revision": n.revision,
                "appareil": n.appareil, "phrases_calibration": len(c.scores)}

    return app


app = creer_app()
