"""La confiance trouve-t-elle les erreurs ? Relecture simulee sur le jeu de test.

L'interface de validation presente les predictions les moins sures d'abord.
Ce classement ne vaut que si les phrases fausses ont bien une confiance plus
basse que les justes. La calibration l'a verifie sur le jeu de developpement,
tire des memes sources que l'entrainement ; ce script le verifie sur les 678
phrases d'evaluation de TerjamaBench, ecrites par des humains.

La relecture est simulee : on lit les phrases de la moins sure a la plus sure,
et l'on compte, a chaque etape, la part des phrases fausses et des erreurs de
caractere deja trouvees. Une lecture au hasard en trouverait la meme part que
de phrases lues ; l'ecart entre les deux est ce que la confiance apporte.

Les predictions viennent du service tel qu'il est servi, quantifie compris :
c'est sa confiance qui ordonne l'interface. Rien n'est regle ici.

Usage, depuis la racine du depot :
    python -m evaluation.relecture --service http://127.0.0.1:8000
    python -m evaluation.relecture                 relit relecture.csv deja ecrit
"""

import argparse

import pandas as pd
from huggingface_hub import hf_hub_download

from donnees.decouper_test import DEPOT, TEST, prelever
from evaluation.evaluate import distance, mesurer, normaliser

FICHIER = "relecture.csv"
PARTS = (0.05, 0.10, 0.20, 0.30, 0.50)
LOT = 64


def par_phrase(references, predictions):
    """Erreurs de caractere, longueur et exactitude de chaque phrase, normalisees."""
    lignes = []
    for r, p in zip(references, predictions):
        r, p = normaliser(str(r)), normaliser(str(p))
        lignes.append({"erreurs": distance(p, r), "longueur": len(r), "exacte": p == r})
    return pd.DataFrame(lignes)


def auroc(confiances, fausses):
    """Probabilite qu'une phrase fausse soit moins sure qu'une juste, ex aequo pour moitie.

    1 : la confiance separe parfaitement ; 0,5 : elle ne vaut pas le hasard.
    """
    d = pd.DataFrame({"c": list(confiances), "f": list(fausses)})
    rangs = d["c"].rank(method="average")
    n_f, n_j = int(d["f"].sum()), int((~d["f"]).sum())
    if not n_f or not n_j:
        return float("nan")
    # Mann-Whitney sur les justes : combien de fausses chacune depasse.
    u = rangs[~d["f"]].sum() - n_j * (n_j + 1) / 2
    return u / (n_f * n_j)


def courbe(donnees, parts=PARTS):
    """Part des fausses et des erreurs trouvees en relisant les moins sures d'abord."""
    d = donnees.sort_values("confiance", kind="stable").reset_index(drop=True)
    fausses, erreurs = (~d["exacte"]).sum(), d["erreurs"].sum()
    lignes = []
    for part in parts:
        lues = d.iloc[:max(1, round(part * len(d)))]
        lignes.append({
            "part": part,
            "phrases relues": len(lues),
            "fausses trouvees": (~lues["exacte"]).sum() / max(fausses, 1),
            "erreurs trouvees": lues["erreurs"].sum() / max(erreurs, 1),
            "taux de fausses relues": (~lues["exacte"]).mean(),
        })
    return pd.DataFrame(lignes)


def interroger(url, textes):
    """Sortie, rang et confiance brute du service, pour chaque texte, dans l'ordre."""
    import httpx

    resultats = []
    with httpx.Client(base_url=url, timeout=600) as client:
        sante = client.get("/sante").json()
        print(f"service : {sante['modele']}@{sante['revision'][:7]} en "
              f"{sante.get('variante', 'fp32')}, registre "
              f"{'actif' if sante.get('registre') else 'inactif'}")
        for i in range(0, len(textes), LOT):
            r = client.post("/normaliser/lot", json={"textes": textes[i:i + LOT]})
            r.raise_for_status()
            resultats += r.json()["resultats"]
    return resultats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--service", help="adresse du service a interroger ; sinon, "
                                     f"relit {FICHIER}")
    a = p.parse_args()

    if a.service:
        test = pd.read_csv(hf_hub_download(DEPOT, f"splits/{TEST}", repo_type="dataset"))
        _, evaluation = prelever(test)
        r = interroger(a.service, evaluation["transliteration"].astype(str).tolist())
        evaluation = evaluation.assign(
            prediction=[x["sortie"] for x in r], rang=[x["confiance"] for x in r],
            confiance=[x["confiance_brute"] for x in r])
        evaluation.to_csv(FICHIER, index=False, encoding="utf-8-sig")
        print(f"{len(evaluation)} predictions ecrites dans {FICHIER}")
    evaluation = pd.read_csv(FICHIER)

    m = mesurer(evaluation["darija_norm"].map(normaliser),
                evaluation["prediction"].map(normaliser))
    d = pd.concat([evaluation.reset_index(drop=True),
                   par_phrase(evaluation["darija_norm"], evaluation["prediction"])], axis=1)
    print(f"\n{len(d)} phrases d'evaluation : CER {m['cer']:.4f}, "
          f"{(~d['exacte']).sum()} fausses ({100 * (~d['exacte']).mean():.1f} %)")
    print(f"AUROC, une fausse moins sure qu'une juste : {auroc(d['confiance'], ~d['exacte']):.3f}")

    print("\nRelecture des moins sures d'abord, contre une relecture au hasard")
    print(f"{'relues':>8}{'phrases':>9}{'fausses trouvees':>18}{'erreurs trouvees':>18}"
          f"{'fausses parmi les relues':>26}")
    for _, x in courbe(d).iterrows():
        print(f"{100 * x['part']:>7.0f}%{x['phrases relues']:>9.0f}"
              f"{100 * x['fausses trouvees']:>17.1f}%{100 * x['erreurs trouvees']:>17.1f}%"
              f"{100 * x['taux de fausses relues']:>25.1f}%")
    print(f"  au hasard, chaque ligne trouverait sa propre part, et "
          f"{100 * (~d['exacte']).mean():.1f} % de fausses parmi les relues")

    print("\nPar decile de confiance, du moins sur au plus sur")
    d["decile"] = pd.qcut(d["confiance"].rank(method="first"), 10, labels=False) + 1
    print(f"{'decile':>7}{'confiance':>11}{'CER':>8}{'exactitude':>12}")
    for dec, g in d.groupby("decile"):
        print(f"{dec:>7}{g['confiance'].median():>11.3f}"
              f"{g['erreurs'].sum() / max(g['longueur'].sum(), 1):>8.4f}"
              f"{100 * g['exacte'].mean():>11.1f}%")


if __name__ == "__main__":
    main()
