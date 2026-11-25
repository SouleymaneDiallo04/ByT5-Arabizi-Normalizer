# syntax=docker/dockerfile:1
# Image du service (ENF-01). Construction, depuis la racine du depot :
#   docker build --secret id=hf_token,env=HF_TOKEN -t byt5-arabizi .
#   docker run --rm -p 8000:8000 byt5-arabizi
# Le modele est telecharge a la construction : le conteneur demarre ensuite
# sans reseau ni jeton.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# torch en version processeur : la roue par defaut embarque CUDA, plusieurs
# gigaoctets inutiles ici. Installee d'abord, elle satisfait ensuite la ligne
# torch==2.12.1 de requirements-service.txt.
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-service.txt .
RUN pip install -r requirements-service.txt

RUN useradd --create-home service

# Le modele, a la revision figee dans service/config.py. Seul ce fichier est
# copie a ce stade : modifier le code ne force pas a retelecharger 1,2 Go. Le
# jeton passe par un secret de construction, qui ne reste dans aucune couche.
# Le chown se fait dans la meme etape : fait a part, il recopiait le modele
# dans une seconde couche, et l'image portait deux fois ses 1,2 Go.
COPY service/__init__.py service/config.py service/
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python -c "\
from huggingface_hub import snapshot_download; \
from service.config import MODELE, REVISION; \
snapshot_download(MODELE, revision=REVISION, allow_patterns=[ \
    'config.json', 'generation_config.json', 'model.safetensors', \
    'tokenizer_config.json', 'added_tokens.json'])" \
 && chown -R service /opt/hf

COPY modele/__init__.py modele/inference.py modele/quantification.py modele/
COPY service/ service/
COPY registre/__init__.py registre/schema.py registre/depot.py registre/

# Tout est dans l'image : interdire le reseau a Hugging Face garantit que le
# conteneur ne depend pas de l'hote, et ferait echouer un fichier manquant au
# demarrage plutot qu'a la premiere requete.
ENV HF_HUB_OFFLINE=1

USER service

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/sante', timeout=4)"

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
