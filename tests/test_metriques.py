"""Le WER, et la reparation des poids partages lors de la comparaison de modeles."""

import math
import sys
import types

import torch

from evaluation.evaluate import mesurer


def test_wer_compte_les_mots_au_niveau_du_corpus():
    m = mesurer(["a b c d", "e"], ["a x c d", "e"])
    assert math.isclose(m["wer"], 1 / 5)
    assert m["exactitude"] == 0.5


def test_wer_insertion_et_suppression():
    assert math.isclose(mesurer(["a b"], ["a x b"])["wer"], 1 / 2)
    assert math.isclose(mesurer(["a b c"], ["a c"])["wer"], 1 / 3)


def test_wer_nul_sur_des_phrases_identiques():
    assert mesurer(["شريت 3 طوموبيلات"], ["شريت 3 طوموبيلات"])["wer"] == 0


def test_une_lettre_fausse_compte_un_mot_entier():
    m = mesurer(["شريت طوموبيلا"], ["شريت طوموبيلة"])
    assert math.isclose(m["wer"], 1 / 2) and m["cer"] < 0.1


def faux_t5():
    class Bloc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 2)

    class Modele(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Embedding(4, 2)
            self.encoder, self.decoder = Bloc(), Bloc()
            self.lm_head = torch.nn.Linear(2, 4, bias=False)

    return Modele()


def importer_comparer():
    if "transformers" not in sys.modules:
        t = types.ModuleType("transformers")
        t.AutoModelForSeq2SeqLM = t.AutoTokenizer = object
        sys.modules["transformers"] = t
    from evaluation.comparer_modeles import reparer
    return reparer


def test_reparer_relie_les_plongements_et_signale_le_reste():
    reparer = importer_comparer()
    m = faux_t5()
    restantes = reparer(m, ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight",
                            "encoder.block.0.autre.weight"])
    assert m.encoder.embed_tokens.weight is m.shared.weight
    assert m.decoder.embed_tokens.weight is m.shared.weight
    assert m.lm_head.weight is not m.shared.weight
    assert restantes == ["encoder.block.0.autre.weight"]


def test_reparer_quand_la_matrice_est_rangee_sous_un_plongement():
    reparer = importer_comparer()
    m = faux_t5()
    restantes = reparer(m, ["shared.weight", "decoder.embed_tokens.weight"])
    assert restantes == []
    assert m.shared.weight is m.encoder.embed_tokens.weight
    assert m.decoder.embed_tokens.weight is m.encoder.embed_tokens.weight


def test_reparer_signale_si_aucune_matrice_n_est_dans_le_fichier():
    reparer = importer_comparer()
    m = faux_t5()
    noms = ["shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight",
            "lm_head.weight"]
    assert sorted(reparer(m, noms)) == sorted(noms)
