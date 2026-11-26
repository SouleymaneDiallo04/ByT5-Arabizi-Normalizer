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
