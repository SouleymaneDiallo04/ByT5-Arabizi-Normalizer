# Feuille de route

## Données

- [x] Extraction de `DrIAmed/darija-youtube-dataset` (2 706 lignes)
- [x] Extraction de `MBZUAI-Paris/DODa-10K` (10 000 lignes)
- [x] Extraction de `OiQ/daa-pairs` (112 814 lignes)
- [x] Extraction de `Haitam03/doda_synthetic` (64 260 lignes)
- [x] Rédaction de la carte du dataset Hugging Face
- [ ] Fusion et nettoyage (`merge_all.py`) et publication de `darija_all.csv`
- [ ] Amorçage d'un lexique français ↔ écriture arabe à partir de la colonne
      `mixte` de `daa-pairs`
- [ ] Liste de mots français fréquents par domaine (santé, technologie,
      administration, banque, école, commerce, etc.)
- [ ] Génération de phrases mixtes darija / français, avec cible en écriture
      arabe et cible mixte
- [ ] Ajout de bruit réaliste sur l'arabizi (apostrophes oubliées, voyelles
      supprimées, chiffres variables)
- [ ] Nouveau Testament en darija : traduction depuis le français (Groq) puis
      validation par un second modèle (Gemini)
- [ ] Jeu de test réel, relu à la main

## Modèle

- [ ] Choix de la configuration de ByT5
- [ ] Entraînement de référence
- [ ] Évaluation (métriques à définir)
- [ ] Analyse d'erreurs

## Documentation

- [x] Structure minimale de la documentation
- [ ] Déploiement sur Read the Docs
- [ ] Pages Modèle et Évaluation
