"""Calculs de evaluation/relecture.py, sur des donnees fabriquees."""

import pandas as pd
import pytest

from evaluation.relecture import auroc, courbe, par_phrase


def test_par_phrase():
    d = par_phrase(["شريت", "واش"], ["شريت", "واخ"])
    assert d["exacte"].tolist() == [True, False]
    assert d["erreurs"].tolist() == [0, 1]
    assert d["longueur"].tolist() == [4, 3]


def test_auroc():
    # Les fausses sont toutes moins sures : separation parfaite.
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 1.0
    # A l'envers : la confiance designerait les justes.
    assert auroc([0.8, 0.9, 0.1, 0.2], [True, True, False, False]) == 0.0
    # Ex aequo partout : le hasard.
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_courbe_relit_les_moins_sures_d_abord():
    d = pd.DataFrame({"confiance": [0.9, 0.1, 0.5, 0.3],
                      "exacte": [True, False, True, False],
                      "erreurs": [0, 6, 0, 2]})
    c = courbe(d, parts=(0.25, 0.5, 1.0)).set_index("part")
    # Le quart le moins sur est la phrase a 0,1 : une fausse sur deux, 6 erreurs sur 8.
    assert c.loc[0.25, "fausses trouvees"] == pytest.approx(0.5)
    assert c.loc[0.25, "erreurs trouvees"] == pytest.approx(0.75)
    assert c.loc[0.5, "fausses trouvees"] == pytest.approx(1.0)
    assert c.loc[0.5, "taux de fausses relues"] == pytest.approx(1.0)
    assert c.loc[1.0, "erreurs trouvees"] == pytest.approx(1.0)
