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
- `baseline.py` : baseline déterministe, table de mots apprise sur le train plus
  conversion caractère par caractère. CER 0,1735 et exactitude phrase 18,2 %
  sur le jeu de test, contre 0,2224 et 9,6 % pour les règles seules
- Score de confiance de la baseline (EF-03) : moyenne des confiances par mot,
  calibree sur `dev.csv`. Un mot de la table est exact dans 87,2 % des cas,
  un mot converti par regles dans 24,7 %. Le score decroit de facon monotone
  avec le CER sur le jeu de test, qu'il n'a jamais vu
- `evaluate.py` : calcul du CER et de l'exactitude phrase par une commande
  unique, ventilé par type, par thème et par variété régionale
- `test_robustesse.py` et `requirements.txt` : contrôles de recette ENF-08 et ENF-04
- Carte du dataset pour Hugging Face
- Structure initiale de la documentation
- Déploiement de la documentation sur Read the Docs
- `.gitignore` : exclusion des CSV et Parquet produits, qui dépassent la limite
  de taille de GitHub
