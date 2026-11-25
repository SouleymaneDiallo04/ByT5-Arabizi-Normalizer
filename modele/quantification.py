"""Quantification dynamique en entier 8 bits, partagee par la mesure et le service.

evaluation/quantification.py a choisi les couches a quantifier, groupe par
groupe (journal du 20 novembre). Le service applique le meme choix : les deux
passent par ce module, et non par deux copies des memes motifs.

Seul torch est requis : le module se teste sans transformers ni modele.
"""

import re

import torch

# Couches lineaires de ByT5, par role. Le decodeur a trois sous-couches par
# bloc (auto-attention, attention croisee, feed-forward), l'encodeur deux.
GROUPES = {
    "enc_attention": r"^encoder\.block\.\d+\.layer\.0\.SelfAttention\.[qkvo]$",
    "enc_ffn_entree": r"^encoder\.block\.\d+\.layer\.1\.DenseReluDense\.wi(_\d)?$",
    "enc_ffn_sortie": r"^encoder\.block\.\d+\.layer\.1\.DenseReluDense\.wo$",
    "dec_attention": r"^decoder\.block\.\d+\.layer\.[01]\."
                     r"(SelfAttention|EncDecAttention)\.[qkvo]$",
    "dec_ffn_entree": r"^decoder\.block\.\d+\.layer\.2\.DenseReluDense\.wi(_\d)?$",
    "dec_ffn_sortie": r"^decoder\.block\.\d+\.layer\.2\.DenseReluDense\.wo$",
    "tete": r"^lm_head$",
}

# Le choix servi : tout sauf la tete, qui double le CER a elle seule, et la
# sortie des feed-forward de l'encodeur, defaut connu de la famille T5. Soit
# 116 couches sur 129, pour un CER de 0,1042 contre 0,1033 en fp32.
GROUPES_SERVIS = ("enc_attention", "enc_ffn_entree", "dec_attention",
                  "dec_ffn_entree", "dec_ffn_sortie")


def couches(modele):
    """Noms des couches lineaires de chaque groupe, et celles hors de tout groupe.

    Leve ValueError si un groupe ne trouve aucune couche : un motif qui ne
    correspond plus aux noms du modele ferait passer une configuration pour
    inoffensive alors qu'elle n'aurait rien quantifie.
    """
    lineaires = [n for n, m in modele.named_modules() if isinstance(m, torch.nn.Linear)]
    par_groupe = {g: [n for n in lineaires if re.match(motif, n)]
                  for g, motif in GROUPES.items()}
    vides = [g for g, noms in par_groupe.items() if not noms]
    if vides:
        raise ValueError(f"groupes sans couche : {vides} ; "
                         f"exemples de noms du modele : {lineaires[:6]}")
    rangees = set().union(*map(set, par_groupe.values()))
    return par_groupe, sorted(set(lineaires) - rangees)


def quantifier(modele, noms):
    """Quantification dynamique en entier 8 bits des seules couches nommees."""
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(modele, set(noms), dtype=torch.qint8)


def quantifier_modele(modele, groupes=GROUPES_SERVIS):
    """Quantifie les groupes retenus ; rend le modele et le nombre de couches."""
    par_groupe, _ = couches(modele)
    noms = [n for g in groupes for n in par_groupe[g]]
    return quantifier(modele, noms), len(noms)
