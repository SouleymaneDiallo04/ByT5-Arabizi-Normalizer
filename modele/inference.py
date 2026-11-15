"""Decodage et score de confiance, partages par les mesures et par le service.

Le service renvoie le rang de la confiance parmi celles du jeu de developpement
(PO-05). Ce rang n'a de sens que si la confiance servie est calculee exactement
comme celle qui a servi a calibrer : les deux passent donc par ce module, et
non par deux copies du meme code.
"""

import torch

# Longueurs maximales en octets, celles de l'entrainement (modele/train_byt5.py).
# Une entree plus longue serait tronquee par le tokenizer : le service la refuse
# plutot que de rendre en silence une phrase amputee.
MAX_SOURCE, MAX_CIBLE = 192, 288
TAILLE_LOT = 16


def confiance_moyenne(transitions, jetons, pad_id):
    """Probabilite moyenne par octet produit, une valeur par phrase.

    `transitions` porte les log-probabilites de chaque jeton genere, `jetons` les
    jetons eux-memes, sans le jeton de depart du decodeur. generate complete
    apres la fin de sequence avec le jeton de remplissage : le compter gonflerait
    la confiance, le modele lui donnant une probabilite proche de 1. Chez ByT5
    seul le remplissage porte l'identifiant 0, les octets commencent a 3.

    La moyenne est prise sur les log-probabilites puis exponentiee : le resultat
    est la moyenne geometrique des probabilites, dans [0,1] par construction
    (EF-03), et comparable entre phrases de longueurs differentes.
    """
    garde = jetons != pad_id
    transitions = transitions.masked_fill(~garde, 0.0)
    moyenne = transitions.sum(-1) / garde.sum(-1).clamp(min=1)
    return moyenne.exp()


def decoder(modele, tok, phrases, appareil, faisceaux=1, confiance=True):
    """Decode une liste de phrases, par lots de TAILLE_LOT et dans l'ordre.

    Renvoie trois listes alignees sur `phrases` : les textes, les confiances
    (vide si `confiance` est faux) et, pour chaque sortie, si le modele l'a
    terminee. Une sortie qui atteint MAX_CIBLE octets sans fin de sequence est
    tronquee : l'appelant doit pouvoir le savoir.
    """
    textes, confiances, completes = [], [], []
    for depart in range(0, len(phrases), TAILLE_LOT):
        lot = [str(p) for p in phrases[depart:depart + TAILLE_LOT]]
        entrees = tok(lot, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SOURCE).to(appareil)
        with torch.no_grad():
            sortie = modele.generate(**entrees, max_length=MAX_CIBLE,
                                     num_beams=faisceaux,
                                     output_scores=confiance,
                                     return_dict_in_generate=True)
        jetons = sortie.sequences[:, 1:]
        textes.extend(tok.batch_decode(sortie.sequences, skip_special_tokens=True))
        completes.extend((jetons == tok.eos_token_id).any(-1).tolist())

        if confiance:
            transitions = modele.compute_transition_scores(
                sortie.sequences, sortie.scores, normalize_logits=True)
            confiances.extend(
                confiance_moyenne(transitions, jetons, tok.pad_token_id).tolist())

    return textes, confiances, completes


def predire(modele, tok, phrases, appareil, faisceaux=1, confiance=False):
    """Decode une liste de phrases, et renvoie eventuellement la confiance.

    Renvoie `(textes, confiances)`, `confiances` valant None si elle n'est pas
    demandee. Forme utilisee par les scripts de mesure.
    """
    textes, confiances, _ = decoder(modele, tok, phrases, appareil,
                                    faisceaux, confiance)
    return textes, (confiances if confiance else None)
