"""Selection des couches quantifiees, sur une copie des noms de ByT5.

Le vrai modele pese 1,2 Go et demande transformers : une arborescence de
modules aux memes noms suffit a verifier quelles couches passent en entier
8 bits, et lesquelles restent en pleine precision.
"""

import pytest
from torch import nn

from modele.quantification import GROUPES, GROUPES_SERVIS, couches, quantifier_modele


def module(**enfants):
    m = nn.Module()
    for nom, enfant in enfants.items():
        m.add_module(nom, enfant)
    return m


def lineaires(*noms):
    return module(**{n: nn.Linear(4, 4, bias=False) for n in noms})


def faux_byt5(blocs_encodeur=12, blocs_decodeur=4, tete=True):
    """Les couches lineaires de T5ForConditionalGeneration, a leurs noms exacts."""
    attention = lambda: lineaires("q", "k", "v", "o")
    ffn = lambda: module(DenseReluDense=lineaires("wi_0", "wi_1", "wo"))
    encodeur = nn.ModuleList(
        module(layer=nn.ModuleList([module(SelfAttention=attention()), ffn()]))
        for _ in range(blocs_encodeur))
    decodeur = nn.ModuleList(
        module(layer=nn.ModuleList([module(SelfAttention=attention()),
                                    module(EncDecAttention=attention()), ffn()]))
        for _ in range(blocs_decodeur))
    parties = {"encoder": module(block=encodeur), "decoder": module(block=decodeur)}
    if tete:
        parties["lm_head"] = nn.Linear(4, 4, bias=False)
    return module(**parties)


def test_groupes_servis_ceux_du_journal():
    """Journal du 20 novembre : tout sauf la tete et la sortie des
    feed-forward de l'encodeur."""
    assert set(GROUPES) - set(GROUPES_SERVIS) == {"tete", "enc_ffn_sortie"}


def test_116_couches_sur_129():
    modele = faux_byt5()
    par_groupe, hors_groupe = couches(modele)
    assert sum(map(len, par_groupe.values())) == 129 and hors_groupe == []
    _, n = quantifier_modele(modele)
    assert n == 116


def test_la_tete_et_la_sortie_des_ffn_de_l_encodeur_restent_en_fp32():
    q, _ = quantifier_modele(faux_byt5(blocs_encodeur=2, blocs_decodeur=1))
    types = {nom: type(m) for nom, m in q.named_modules()}
    assert types["lm_head"] is nn.Linear
    assert types["encoder.block.1.layer.1.DenseReluDense.wo"] is nn.Linear
    for nom in ("encoder.block.0.layer.0.SelfAttention.q",
                "encoder.block.1.layer.1.DenseReluDense.wi_0",
                "decoder.block.0.layer.1.EncDecAttention.o",
                "decoder.block.0.layer.2.DenseReluDense.wo"):
        assert types[nom] is not nn.Linear, nom
        assert "quantized" in types[nom].__module__, nom


def test_un_groupe_introuvable_arrete():
    """Un motif qui ne correspond plus aux noms ne doit pas passer pour
    inoffensif alors qu'il n'aurait rien quantifie."""
    with pytest.raises(ValueError, match="tete"):
        quantifier_modele(faux_byt5(tete=False))
