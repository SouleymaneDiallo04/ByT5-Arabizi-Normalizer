"""Fine-tuning de ByT5 pour la normalisation arabizi vers ecriture arabe.

L'entree est la colonne transliteration, la cible la colonne darija_norm, c'est
a dire la cible ramenee a la convention retenue. Le modele est selectionne sur
dev.csv par arret precoce, puis applique a test_real.csv. Les predictions sont
ecrites au format attendu par evaluate.py, de sorte que le modele et la baseline
soient mesures par la meme commande.

Ce script demande un GPU. Sur processeur seul, une epoque sur 186 861 paires
depasse la journee : le decodage est autoregressif et travaille a l'octet.

Usage : python train_byt5.py [nombre_de_paires_pour_un_essai_rapide]
"""

import sys

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          set_seed)

MODELE = "google/byt5-small"
TRAIN, DEV, TEST = "train.csv", "dev.csv", "test_real.csv"
SORTIE_MODELE = "byt5-arabizi"
SORTIE_PREDICTIONS = "byt5_predictions.csv"

# Longueurs fixees sur la distribution reelle des octets du corpus : l'entree
# ne depasse jamais 178 octets, la cible 280. Ces valeurs couvrent donc 100 %
# des paires sans troncature, en gaspillant le moins de remplissage possible.
MAX_SOURCE = 192
MAX_CIBLE = 288

GRAINE = 42
EPOQUES = 3
TAILLE_LOT = 16
PAS_ACCUMULATION = 2
TAUX_APPRENTISSAGE = 3e-4
PATIENCE = 2
# Decodage glouton : PO-06 doit etre arbitre sur mesure reelle, et la recherche
# en faisceau multiplie une latence deja contrainte par le decodage a l'octet.
FAISCEAUX = 1


class Corpus(Dataset):
    """Paires arabizi vers arabe, encodees en octets pour ByT5."""

    def __init__(self, donnees, tokenizer):
        self.entrees = donnees["transliteration"].astype(str).tolist()
        self.cibles = donnees["darija_norm"].astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.entrees)

    def __getitem__(self, i):
        encode = self.tokenizer(self.entrees[i], max_length=MAX_SOURCE,
                                truncation=True)
        encode["labels"] = self.tokenizer(self.cibles[i], max_length=MAX_CIBLE,
                                          truncation=True)["input_ids"]
        return encode


def charger(chemin, echantillon=None):
    """Lit un fichier de paires, eventuellement reduit pour un essai rapide."""
    d = pd.read_csv(chemin)
    return d.head(echantillon) if echantillon else d


if __name__ == "__main__":
    echantillon = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if not torch.cuda.is_available():
        print("ATTENTION : aucun GPU detecte, l'entrainement sera inexploitable.")

    set_seed(GRAINE)
    tokenizer = AutoTokenizer.from_pretrained(MODELE)
    modele = AutoModelForSeq2SeqLM.from_pretrained(MODELE)

    train = charger(TRAIN, echantillon)
    dev = charger(DEV, echantillon and max(echantillon // 10, 1))
    test = charger(TEST)
    print(f"{len(train)} paires d'entrainement, {len(dev)} de validation, "
          f"{len(test)} de test")

    arguments = Seq2SeqTrainingArguments(
        output_dir=SORTIE_MODELE,
        seed=GRAINE,
        data_seed=GRAINE,
        num_train_epochs=EPOQUES,
        per_device_train_batch_size=TAILLE_LOT,
        per_device_eval_batch_size=TAILLE_LOT,
        gradient_accumulation_steps=PAS_ACCUMULATION,
        learning_rate=TAUX_APPRENTISSAGE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=200,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )

    entraineur = Seq2SeqTrainer(
        model=modele,
        args=arguments,
        train_dataset=Corpus(train, tokenizer),
        eval_dataset=Corpus(dev, tokenizer),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=modele),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    entraineur.train()
    entraineur.save_model(SORTIE_MODELE)
    tokenizer.save_pretrained(SORTIE_MODELE)

    # Predictions sur le jeu de test, au format attendu par evaluate.py
    modele.eval()
    appareil = modele.device
    predictions = []
    for depart in range(0, len(test), TAILLE_LOT):
        lot = test["transliteration"].astype(str).iloc[depart:depart + TAILLE_LOT].tolist()
        entrees = tokenizer(lot, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_SOURCE).to(appareil)
        with torch.no_grad():
            sorties = modele.generate(**entrees, max_length=MAX_CIBLE,
                                      num_beams=FAISCEAUX)
        predictions.extend(tokenizer.batch_decode(sorties, skip_special_tokens=True))

    resultat = test[["id", "transliteration", "darija", "darija_norm",
                     "topic", "annotator_dialect", "mot_isole"]].copy()
    resultat["prediction"] = predictions
    resultat.to_csv(SORTIE_PREDICTIONS, index=False, encoding="utf-8-sig")
    print(f"{len(resultat)} predictions enregistrees dans {SORTIE_PREDICTIONS}")
    print(f"mesurer avec : python evaluate.py {SORTIE_PREDICTIONS}")
