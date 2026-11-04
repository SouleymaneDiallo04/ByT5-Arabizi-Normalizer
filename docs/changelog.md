# Journal des modifications

## Non publié

- Extraction de cinq sources de données : darija-youtube-dataset, DODa-10K,
  daa-pairs, doda_synthetic, DODa-audio-dataset-V3
- `merge_all.py` : fusion des cinq fichiers en `darija_all.csv` (196 770 lignes),
  avec retrait des lignes vides et des doublons de paires, et marquage des
  cibles divergentes et des paires mal alignées
- `make_splits.py` : jeu de test tiré de TerjamaBench, purge des 110 lignes du
  corpus qui le recoupaient, cible normalisée et découpage `train` / `dev` sur
  la clé de phrase pour éviter toute fuite entre les blocs
- Interdiction consignée du fichier `test` d'Atlaset comme matière première :
  830 de ses lignes sont celles du jeu de test
- Carte du dataset pour Hugging Face
- Structure initiale de la documentation
- Déploiement de la documentation sur Read the Docs
- `.gitignore` : exclusion des CSV et Parquet produits, qui dépassent la limite
  de taille de GitHub
