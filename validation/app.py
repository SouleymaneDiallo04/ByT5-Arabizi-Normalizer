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
