# Journal des modifications

## Non publié

- Extraction de cinq sources de données : darija-youtube-dataset, DODa-10K,
  daa-pairs, doda_synthetic, DODa-audio-dataset-V3
- `merge_all.py` : fusion des cinq fichiers en `darija_all.csv` (196 770 lignes),
  avec retrait des lignes vides et des doublons de paires, et marquage des
  cibles divergentes et des paires mal alignées
- Carte du dataset pour Hugging Face
- Structure initiale de la documentation
- Déploiement de la documentation sur Read the Docs
- `.gitignore` : exclusion des CSV et Parquet produits, qui dépassent la limite
  de taille de GitHub
