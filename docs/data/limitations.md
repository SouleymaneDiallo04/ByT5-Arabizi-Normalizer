# Limites et précautions

- **Conventions d'arabizi différentes** selon les sources. La colonne `source`
  permet de les séparer ou de les pondérer avant l'entraînement.
- **Données synthétiques.** `daa-pairs` et `doda_synthetic` ont été produites ou
  corrigées par des modèles de langage, sans relecture exhaustive par des
  locuteurs natifs. Elles peuvent contenir des erreurs. Il est conseillé de les
  traiter à part de `DODa-10K` pour l'évaluation.
- **Doublons entre sources** possibles au-delà des doublons exacts retirés dans
  `darija_all.csv`.
- **Pas de jeu de test réel.** Le corpus ne contient ni messages mixtes
  (darija et français dans la même phrase) ni vrais messages de messagerie.
- **Licences hétérogènes.** Voir {doc}`sources`. Le corpus fusionné est publié
  sous CC BY-NC 4.0, la plus restrictive des licences connues.
