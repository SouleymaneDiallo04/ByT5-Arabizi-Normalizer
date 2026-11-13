pour demarer il faut docn se positionner ici :
cd _build\html
python -m http.server 8000

mais avant cela il faut deja etre dans le document cd docs


et ell est accesible au navigateur a travers donc le lien suivant:http://localhost:8000




# 1. Se placer au bon endroit (dans le dossier docs)
cd /d "C:\Users\Alif computer\Desktop\projet technologie web\Proejt_Expertise\ByT5-Arabizi-Normalizer\docs"

# 2. Forcer une reconstruction complète et propre de la doc
sphinx-build -E -a -b html . _build/html