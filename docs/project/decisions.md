# Journal des décisions

Le cahier des charges fixe les décisions d'arbitrage PO-01 à PO-06 et n'est plus
modifié. Ce journal consigne ce que la mesure a établi depuis, y compris les
résultats négatifs — une spécification dit ce qu'on fait, pas ce qu'on a essayé
sans succès, et c'est pourtant ce second registre qui explique le premier.

Chaque entrée dit ce qui a été décidé, sur quelle mesure, et ce que ça entraîne.
Les plus récentes d'abord.

---

## 26 novembre 2026 — La confiance dit surtout ce qu'on peut relire en dernier

**Question.** L'interface de validation présente les prédictions les moins sûres
d'abord. Ce classement trouve-t-il les erreurs ? La calibration l'avait vérifié
sur le dev, tiré des mêmes sources que l'entraînement. Ici, on le vérifie sur
les 678 phrases d'évaluation de TerjamaBench, avec la confiance du service tel
qu'il est servi, en int8 : `python -m evaluation.relecture --service URL`. La
relecture est simulée, de la phrase la moins sûre à la plus sûre. Rien n'est
réglé.

**Mesure.** 466 phrases sur 678 sont fausses, soit 68,7 %.

| | AUROC |
|---|---:|
| les 678 phrases | 0,691 |
| sans les 130 mots isolés | 0,776 |
| les mots isolés seuls | 0,658 |

L'AUROC est la probabilité qu'une phrase fausse soit moins sûre qu'une phrase
juste : 0,5 pour le hasard, 1 pour une séparation parfaite.

| sans les mots isolés, 548 phrases | fausses trouvées | erreurs trouvées | fausses parmi les relues |
|---|---:|---:|---:|
| relire les 10 % les moins sûres | 11,9 % | 13,1 % | 92,7 % |
| relire les 20 % les moins sûres | 24,0 % | 25,7 % | 93,6 % |
| relire la moitié la moins sûre | 58,4 % | 64,8 % | 91,6 % |
| au hasard | la part relue | la part relue | 78,5 % |

**Lecture.** Quand près de quatre phrases sur cinq sont fausses, presque toute
relecture en trouve : le haut de la liste fait mieux que le hasard, mais de peu.
La confiance vaut surtout par le bas. Le décile le plus sûr est exact à 77,9 %,
avec un CER de 0,029. Ne pas relire les 10 % les plus sûres ne fait manquer que
3,5 % des phrases fausses et 1,6 % des erreurs de caractère, et le double pour
20 %.

Les mots isolés expliquent une anomalie : ils forment 40 % du décile le moins
sûr, plus souvent exact que les trois suivants. Un mot seul reçoit une confiance
basse et se trouve juste ou faux d'un bloc. TerjamaBench en contient 19 % ; un
service utilisé pour de vraies phrases en recevra sans doute moins.

**Ce que ça entraîne.** L'ordre de l'interface est gardé. La directive veut que
toute prédiction soit validée, et la confiance ne dispense de rien. Elle dit
seulement ce qui peut attendre quand le temps manque : les phrases les plus
sûres.

---

## 26 novembre 2026 — Le service sert le modèle quantifié

**Décision.** La décision du 20 novembre est appliquée : le service charge le
mélange 3, puis quantifie en entier 8 bits 116 de ses 129 couches linéaires. La
tête de sortie et la sortie des feed-forward de l'encodeur restent en pleine
précision. La sélection vit dans `modele/quantification.py`, que la mesure et le
service partagent. `QUANTIFIE=0` sert la pleine précision.

**Vérifié à travers l'API**, sur la même machine, conteneur du service, 678
phrases d'évaluation puis 100 phrases envoyées une à une :

| | fp32 | int8 |
|---|---:|---:|
| CER | 0,1033 | 0,1047 |
| exactitude | 32,2 % | 31,3 % |
| latence médiane d'une phrase | 2,10 s | 1,22 s |
| p90 | 3,64 s | 2,03 s |

Le fp32 retrouve exactement ses chiffres, ce qui valide la comparaison. L'int8
est à 0,1047 contre 0,1042 sur Kaggle : les noyaux entiers diffèrent d'un
processeur à l'autre. La latence baisse de 42 % en médiane et de 44 % au p90,
pour six phrases exactes de moins sur 678.

**La calibration est refaite sur le modèle quantifié**, dans l'image du service,
sur les mêmes 2 000 phrases du dev : le fichier est identique entre les
révisions `ab17817` et `422d77c` du corpus. La quantification déplace les
confiances, médiane 0,9893 contre 0,9906, minimum 0,8752 contre 0,9036. Lues
avec la table du fp32, elles seraient classées 0,06 trop bas en médiane, et 245
phrases du dev au lieu de 190 tomberaient sous le rang 0,10 : 29 % de relectures
en trop. La confiance reste un bon indicateur, avec un CER de 0,125 au premier
décile et de 0,014 au dernier (Spearman +0,388). La calibration porte désormais
sa variante, et le service refuse de démarrer sur la table d'une autre variante
comme d'une autre révision. Le registre enregistre la variante avec la révision.

**La mémoire.** Mémoire résidente du processus du service, lue dans
`/proc/1/status`, après la même charge : 100 requêtes d'une phrase, puis 10 lots
de 64.

| | au démarrage | après la charge | pic |
|---|---:|---:|---:|
| fp32 | 368 Mo | 1 685 Mo | 1 825 Mo |
| int8, première version | 1 852 Mo | 2 045 Mo, après 1 000 requêtes et 10 lots | 2 983 Mo |
| int8, avec `gc.collect()` | 1 156 Mo | 1 370 Mo | 3 009 Mo |

La première version gardait le modèle en pleine précision en mémoire. La
quantification travaille sur une copie, et l'original, dont les modules se
référencent entre eux, n'était pas libéré tant que le ramasse-miettes ne passait
pas. Un `gc.collect()` après la quantification retire 700 Mo au démarrage, et le
service quantifié occupe alors 19 % de moins que le fp32 en régime établi. Il ne
fuit pas : sur 1 000 requêtes, la mémoire n'a pris que 44 Mo, par paliers. Le
défaut qui reste est le pic du démarrage, 3,0 Go, quand original et copie
coexistent : il tient dans les 3,75 Gio de Docker Desktop, sans marge.
Quantifier en place évite la copie mais garde le fichier des poids projeté en
mémoire, 1 580 Mo juste après le chargement.

Le premier relevé, fait avec `docker stats`, donnait 0,9 Gio au fp32 et 2,1 au
int8 : il compte la mémoire du conteneur autrement, et inversait le constat.

**Pour un autre modèle.** Le code ne dépend pas du mélange 3, mais deux choses
en dépendent et seront à refaire si un modèle meilleur le remplace : la
calibration, environ 45 minutes de processeur (README, section Service), et la
confirmation du choix des couches par
`python -m evaluation.quantification --final enc_attention,enc_ffn_entree,dec_attention,dec_ffn_entree,dec_ffn_sortie`.

---

## 25 novembre 2026 — Des paires tirées de darija en écriture arabe n'apporteraient presque rien de ce qui manque

**Question.** `lid-dataset-base` et `fw-darija` offrent des centaines de
milliers de phrases de darija en écriture arabe, sans arabizi en face. En
fabriquer l'arabizi en ferait des paires d'entraînement, comme celles de
NileChat. Avant d'y consacrer du GPU, deux questions : ces textes contiennent-ils
ce qui manque au modèle, et peut-on fabriquer sans GPU un arabizi qui ressemble
à celui des humains ? `python -m evaluation.textes_darija byt5_melange3.csv`.

**Ce qui manque au modèle.** Sur les 850 phrases du test, le mélange 3 rate
1 179 mots de la référence. 199, soit 16,9 %, n'apparaissent ni dans le corpus
du projet ni dans NileChat, et 26,2 % manquent au seul corpus du projet. Le
modèle n'ayant vu qu'un tirage de NileChat, la part réelle est entre les deux.
Les autres mots faux, de 73,8 à 83,1 %, ont été vus à l'entraînement et sont
mal écrits : voyelles, découpage, variantes. Des textes nouveaux ne les visent
pas.

**Ce que les textes en contiennent.**

| | mots faux | jamais vus | présents dans les textes | trois fois ou plus |
|---|---:|---:|---:|---:|
| tout le test | 1 179 | 199 | 135 | 106 |
| Guelmim | 27 | 9 | 5 | 2 |
| Tetouan | 24 | 6 | 3 | 3 |

Si le modèle apprenait chacun des 135 mots présents, il corrigerait 11,5 % de
ses mots faux. C'est un plafond, qu'aucun entraînement n'atteint. Les formes qui
manquent à Guelmim, `تكيم`, `نكيسو`, `الوثة`, n'apparaissent nulle part.

**Volume.** Environ 1,05 million de phrases utilisables, c'est-à-dire courtes,
portant un marqueur du darija et absentes du corpus comme de NileChat :
815 307 de `fw-darija`, 152 246 d'Atlaset, 69 779 de Darija-LID. `fw-darija`
mêle beaucoup d'arabe standard, des prix aux fatwas. Le volume n'est de toute
façon pas ce qui manque : le mélange 3 n'a vu qu'environ 400 000 des 3,28
millions de paires NileChat.

**Arabizi fabriqué sans GPU**, comparé mot à mot à l'arabizi humain du test.

| mots | méthode | CER | identiques |
|---|---|---:|---:|
| connus | graphie la plus fréquente de nos paires | 0,173 | 56,8 % |
| inconnus | règles lettre à lettre | 0,298 | 19,7 % |

La première ligne mesure la variation entre scripteurs, que nul générateur ne
supprime. Les règles s'en écartent sur deux points que l'écriture arabe ne dit
pas : les voyelles brèves (`tlfn` pour `talfan`) et le ك prononcé g (`kbil`
pour `gbil`), qui est justement un trait régional.

**Décision.** Pas de paires fabriquées à partir de ces textes pour l'instant :
le plafond est bas, et les variétés qu'on voulait aider ne sont pas couvertes.
La piste de données reste `moul_sout`, qui apporte de vraies paires et non un
arabizi à fabriquer. AtlasIA a accordé l'accès au jeu de 71 112 paires le
25 novembre au soir ; celui de 35 971 paires est encore en attente. Sa fiche ne
dit ni sa provenance ni sa licence.

**Limites.** Le test sert ici de diagnostic, comme pour le budget d'erreur :
rien n'y est choisi. Un mot présent dans `fw-darija` peut y être de l'arabe
standard, `فبها` y figure 351 fois. Le filtre des marqueurs du darija est
sévère et écarte aussi de bonnes phrases sans marqueur.

---

## 25 novembre 2026 — La validation humaine est en place

**Décision.** La directive *Score de confiance et boucle de feedback* est
implémentée telle que conçue le 16 novembre : chaque prédiction du service est
enregistrée dans une base Postgres dédiée avant que la réponse parte, une
interface la fait confirmer, corriger ou écarter, et
`python -m registre.exporter` rend les corrections au format d'entraînement.

**Ce qui a été tranché en cours de route.**

- Base injoignable : **503**, et non une prédiction rendue sans être
  enregistrée, qui échapperait à la validation sans que rien ne le signale.
- Postgres **17.12** et non 18 : la version 18 a déplacé son répertoire de
  données, et un volume monté à l'ancien chemin ne retiendrait rien.
- La normalisation des cibles et les clés de rapprochement passent dans
  `donnees/normalisation.py`, sans dépendance : l'image de validation n'a ni
  pandas ni torch. `make_splits.py` les réexporte, et rien ne change pour les
  autres scripts.
- L'export retire les phrases du jeu de test avec les clés de `make_splits.py`
  et affiche leur nombre. Une phrase de TerjamaBench soumise au service puis
  corrigée fausserait sinon l'évaluation.

**Vérifié.** 54 tests sur SQLite, sans modèle ni serveur. Avec
`docker compose`, une correction survit au redémarrage des conteneurs, l'arrêt
de la base fait rendre 503 au service, qui rend 200 dès qu'elle revient, et
l'export d'une correction écrit une ligne. L'image de validation n'a pas torch.

**Limite mesurée.** Base arrêtée, le 503 arrive en 15 à 18 s pour la première
requête, puis en 8 s. Ce n'est pas le `connect_timeout` de 5 s : le DNS de
Docker met 8 s à répondre que le nom d'un conteneur arrêté n'existe pas, et ce
délai précède la connexion. Raccourcir le délai du résolveur dans le conteneur
(`dns_opt`) n'y change rien, l'attente étant dans le DNS de Docker lui-même.
Aucune prédiction n'échappe au registre pour autant : seule la réponse d'échec
est lente.

---

## 25 novembre 2026 — Les hamzas ne sont pas neutralisées au calcul des métriques

**Constat.** PO-01 annonce qu'au calcul des métriques, et là seulement, les
hamzas sont neutralisées dans la référence comme dans la prédiction. La commande
de mesure ne l'a jamais fait. `normaliser` dans `evaluation/evaluate.py` retire
la vocalisation, convertit `?` et `;` en signes arabes et réduit les espaces,
rien d'autre. Le remplacement de أ, إ et آ par ا n'existe que dans `cle_arabe`,
qui sert à détecter doublons et fuites entre les jeux, jamais à noter. Tous les
chiffres publiés depuis le premier modèle sont donc **stricts** : une hamza qui
diffère compte comme une erreur.

**Mesure.** Mêmes prédictions, mêmes 678 phrases d'évaluation (`prelever`,
`donnees/decouper_test.py`), même `normaliser`, auquel on ajoute des deux côtés
le remplacement de أ, إ et آ par ا :

| | CER | WER | Exactitude |
|---|---:|---:|---:|
| mélange 3, strict | 0,1033 | 38,9 % | 32,2 % |
| mélange 3, hamzas neutralisées | 0,0989 | 37,5 % | 33,9 % |
| baseline, stricte | 0,1699 | 50,4 % | 17,6 % |
| baseline, hamzas neutralisées | 0,1666 | 49,6 % | 18,7 % |

93 références sur 678 portent une hamza. **12 phrases** du mélange 3 ne sont
fausses que par elle, et le plus souvent le modèle y écrit la forme standard
quand la référence ne le fait pas : ألف ريال pour الف ريال, إنستغرام pour
انستغرام.

**Décision.** La mesure stricte reste la mesure principale, pour trois raisons :
- tous les chiffres déjà présentés et publiés le sont sous cette forme ;
- c'est la plus sévère pour le modèle ;
- l'écart avec la baseline change à peine : 14,6 points d'exactitude en
  mesure stricte, 15,2 en neutralisée.

La version neutralisée est donnée en mesure secondaire, comme la table de
DODa-audio. Le cahier des charges n'étant plus modifié, c'est cette entrée qui
fait foi sur ce point : ses chiffres de PO-01 (+0,0034 de CER, +1,3 point
d'exactitude), mesurés sur le premier modèle, décrivent cette mesure
secondaire et non la principale.

---

## 25 novembre 2026 — Devant tous les modèles publics mesurés

**Mesure.** `evaluation/comparer_modeles.py` fait passer les modèles publics de
translittération sur les mêmes phrases de TerjamaBench, par la même mesure.
Mode d'appel choisi sur les 172 phrases de réglage, chiffres sur les 678
d'évaluation :

| système | domaine visé | CER | WER | exactitude |
|---|---|---:|---:|---:|
| **mélange 3 (nous)** | darija marocain | **0,1033** | **38,9 %** | **32,2 %** |
| akachkach, ByT5 | darija | 0,2477 | 48,1 % | 24,3 % |
| baseline, dictionnaire et règles | darija marocain | 0,1699 | 50,4 % | 17,6 % |
| Haitam03, mT5 | darija marocain | 0,3121 | 65,4 % | 10,0 % |
| AtlasIA, Transformer, mot par mot | darija marocain | 0,2929 | 69,0 % | 9,9 % |
| masterofaudio, ByT5, mot par mot | arabe du Golfe | 0,2499 | 73,2 % | 9,3 % |
| Khenafif, mBART | darija algérien | 0,6055 | 93,0 % | 0,4 % |

Le mT5 de Haitam03, entraîné sur `doda_synthetic`, une de nos sources, a été
mesuré le 26 novembre avec son préfixe documenté, `transliterate_latin_darija: `.
Il reste loin derrière, à 0,3867 de CER sur le réglage et 0,3121 sur
l'évaluation, trois fois le nôtre.

**Trois pièges, et ce qu'ils auraient fait conclure.**

- **Le modèle d'AtlasIA est entraîné sur des mots.** Appelé sur des phrases
  entières, il paraissait à 0,78 de CER ; mot par mot et en minuscules, il est
  à 0,30 sur le réglage. Son tokenizer ne distingue pas une coupure dans un mot
  d'une limite de mots, et il perd des mots, `daba` par exemple.
- **Les poids partagés.** Un T5 sauvegardé par transformers 4 avec
  `tie_word_embeddings=False` n'écrit ses plongements qu'une fois, et
  transformers 5 réinitialise ceux qu'il ne trouve pas : le modèle aurait
  produit du charabia. Le script relie ces matrices, et lit les clés dans le
  fichier lui-même plutôt que dans le rapport de chargement.
- **Le préfixe d'entrée.** Deux modèles n'en documentent aucun : le meilleur de
  six essais est gardé, et leur score peut être un peu sous-estimé.

**Ce que ça entraîne.** Nous ne sommes pas passés à côté d'un modèle public
déjà meilleur. Le score d'AtlasIA ne tranche pas la question d'un petit
Transformer dédié : il est plombé par son tokenizer et son entraînement sur des
mots isolés, sans contexte.

---

## 22 novembre 2026 — Le WER rejoint les deux métriques imposées

**Constat.** Le cadrage impose d'optimiser l'exactitude de phrase et le CER. Les
deux se sont contredits sur le modèle prolongé : meilleur CER, moins de phrases
exactes. Et le problème que décrit le cadrage est un vocabulaire fragmenté, or
aucune des deux ne mesure au niveau du mot : l'exactitude perd toute une phrase
pour une lettre, le CER dilue un mot entièrement faux.

**Décision.** Les trois métriques sont rapportées, et **le WER départage deux
modèles quand l'exactitude et le CER se contredisent.**

| 678 phrases d'évaluation | baseline | mélange 3 |
|---|---:|---:|
| exactitude de phrase | 17,6 % | 32,2 % |
| CER | 0,1699 | 0,1033 |
| WER | 0,5042 | 0,3891 |

**Écartées après mesure.** La part de mots ramenés à une seule forme est la plus
proche de l'objectif, mais elle ne repose que sur 80 mots du jeu de test, et un
modèle qui ramènerait toutes les variantes vers une forme fausse y serait
parfait. L'exactitude à la convention près redéfinit ce qui est juste, et reste
un outil de diagnostic. Le taux de lettres dans le bon alphabet est déjà à
99,9 %. La latence n'est pas une métrique à optimiser : c'est une contrainte du
cadrage, vérifiée sur le modèle final.

---

## 21 novembre 2026 — DODa-audio n'apporte que 360 paires, et non 3 668

**Correction.** Le journal du 12 novembre annonçait 3 668 paires nouvelles, et
leur intégration était attendue comme un apport au corpus. Ce chiffre est faux.
Il soustrayait les 8 697 paires de la V3 *restées sous son nom* après la fusion
aux 12 365 paires du nouveau dépôt. Or la fusion avait retiré de la V3 ses
doublons avec DODa-10K et `doda_synthetic` : ces paires étaient déjà dans le
corpus, sous le nom d'une autre source.

| les 12 365 paires distinctes de DODa-audio | paires |
|---|---:|
| déjà dans la V3 | 11 505 |
| nouvelles par rapport à la V3, mais déjà dans le train via `doda_synthetic` | 338 |
| idem, via DODa-10K | 95 |
| même paire déjà dans le dev | 9 |
| partagent un côté avec le dev, donc interdites à l'entraînement | 16 |
| partagent un côté avec le test, donc interdites à l'entraînement | 42 |
| **vraiment nouvelles** | **360** |

Ces 360 paires font 0,19 % de l'entraînement, ont deux mots en médiane, et 207
d'entre elles sont marquées `conflit`.

**Décision.** Pas de nouvelle version du corpus pour elles seules, et pas
d'entraînement : un tel apport serait indétectable, l'expérience précédente
montrant qu'un écart bien plus grand reste dans le bruit. Elles seront intégrées
avec la prochaine vraie source.

**Ce que ça entraîne.** Une source s'ajoute désormais par
`donnees/ajouter_source.py`, et plus en relançant `make_splits.py`, qui tire le
dev parmi toutes les phrases : une seule ligne ajoutée renouvellerait tout le
dev, ferait évaluer les modèles existants sur du déjà vu, et rendrait les pertes
de validation incomparables. Le nouvel outil recopie le train, garde le dev et le
test intacts, et rend compte de chaque ligne écartée.

---

## 21 novembre 2026 — Prolonger le mélange 3 : un gain trop faible pour être établi

**Question.** La perte de validation du mélange 3 baissait encore à son dernier
contrôle. Le modèle gagne-t-il à apprendre plus longtemps ?

**Expérience.** Depuis les poids du mélange 3 au pas 15 000, sur le même fichier
de 960 000 lignes : 7 500 pas de plus, optimiseur neuf, taux 1e-4, échauffement
de 5 %, décroissance linéaire jusqu'à zéro, graine de tirage décalée. Soit 5 h 43
de GPU. C'est un nouveau cycle de taux et non un run de 22 500 pas d'un seul
tenant. La reprise complète du Trainer a d'abord été essayée : les poids se
rechargeaient, mais Adam fusionné refusait le premier pas.

| | mélange 3 | prolongé |
|---|---:|---:|
| perte de validation | 0,1155 | 0,1136 |
| réglage, 172 phrases : CER | 0,1179 | 0,1119 |
| réglage : phrases exactes | 51 | 51 |
| évaluation, 678 phrases : CER | 0,1033 | 0,1004 |
| évaluation : phrases exactes | 218 | 214 |
| évaluation : phrases avec latin | 1 | 1 |

**Test apparié**, phrase par phrase, test du signe :

| | exactes gagnées / perdues | p | phrases meilleures / pires | p |
|---|---:|---:|---:|---:|
| réglage | 4 / 4 | 1,00 | 20 / 14 | 0,39 |
| évaluation | 8 / 12 | 0,50 | 63 / 54 | 0,46 |

**Lecture.** Le CER baisse sur les deux jeux, mais aucun des deux écarts, ni sur
l'exactitude ni sur le sens de variation des phrases, ne se distingue du hasard.
Le gain est réparti sur presque toutes les catégories d'erreurs, de 2126 à 2056
opérations d'édition sur les 850 phrases : voyelles longues −26, autres
consonnes −16, espaces −15, clitiques −6. C'est la signature d'un effet de durée,
pas d'un mécanisme corrigé. La perte qui se stabilise en fin de cycle (0,1138
puis 0,1136) ne prouve pas un plafond, le taux y arrivant à zéro.

**Décision.** La règle fixée avant la mesure est remplie : perte de phrases
exactes non significative, plus de phrases améliorées que dégradées, latin
stable. Le modèle prolongé devient donc le candidat à livrer. Mais cette règle
était faible, puisqu'elle n'exigeait pas que le gain soit significatif, et il ne
l'est pas. **Il n'est donc pas republié pour l'instant** : publier impose de
refaire la calibration et la mesure de quantification, un travail à faire une
fois, pour un modèle nettement meilleur. Le service reste sur le mélange 3.

**Ce que ça entraîne.** Allonger l'entraînement seul n'est pas le levier
principal : la priorité passe aux données. À budget égal de 15 000 pas, les
prochaines recettes se comparent au mélange 3.

La ventilation montre aussi une asymétrie : le modèle écrit trop d'espaces (179
en trop, 80 manquants), trop de ل collés ou isolés (129 contre 74) et trop de ا
(318 contre 178). Elle a été relevée sur les 850 phrases : toute correction qui
la viserait devra être réglée sur les 172 phrases de réglage seulement.

---

## 20 novembre 2026 — La quantification est retenue, sans sa tête ni la sortie des feed-forward de l'encodeur

**Ce que ça renverse.** Le cahier des charges avait écarté la quantification :
l'entier 8 bits sur toutes les couches linéaires multipliait le CER par 3,8.
Le cadrage du projet demande pourtant un modèle quantifié et à faible latence.
Plutôt que de juger la méthode d'un bloc, on a cherché où se logeait la
dégradation (`evaluation/quantification.py`).

**Sélection**, groupe par groupe, sur les 172 phrases de réglage, sur processeur :

| groupe quantifié seul | couches | CER / fp32 |
|---|---:|---:|
| attention de l'encodeur | 48 | 1,00 |
| entrée des feed-forward de l'encodeur | 24 | 1,00 |
| **sortie des feed-forward de l'encodeur** | 12 | **1,19** |
| attention du décodeur | 32 | 1,01 |
| entrée des feed-forward du décodeur | 8 | 1,02 |
| sortie des feed-forward du décodeur | 4 | 0,95 |
| **tête de sortie (`lm_head`)** | 1 | **2,03** |

Deux groupes portent toute la dégradation. La tête de sortie, une seule couche,
double le CER : c'est elle qui choisit chaque octet, et la moindre perte de
précision s'y lit directement. La sortie des feed-forward de l'encodeur est le
défaut connu de la famille T5. La même couche dans le décodeur, elle, supporte
l'entier 8 bits. Le 0,95 du décodeur est du bruit sur 172 phrases.

Une hypothèse a été contredite au passage : quantifier « tout sauf la sortie
des feed-forward » restait à ×2,03, parce que la tête y était incluse. C'est le
découpage par groupe qui a désigné les vrais coupables.

**Mesure finale**, une seule fois, sur les 678 phrases d'évaluation :

| | taille | CER | exactitude | latence médiane | p90 |
|---|---:|---:|---:|---:|---:|
| baseline | — | 0,1699 | 17,6 % | — | — |
| fp32 | 1 199 Mo | 0,1033 | 32,2 % | 0,990 s | 1,913 s |
| **quantifié, 116 couches sur 129** | **493 Mo** | **0,1042** | **31,4 %** | **0,616 s** | **1,138 s** |

**Décision.** Le service servira le modèle quantifié : 2,4 fois plus léger,
38 % plus rapide en médiane et 40 % au p90, pour +0,9 % de CER relatif et
−0,8 point d'exactitude, soit environ cinq phrases sur 678. La latence est
mesurée sur deux cœurs Kaggle, plus lents que les quatre de la mesure du
16 novembre ; seul le rapport entre les deux lignes compte.

**Ce que ça entraîne.** La confiance du modèle quantifié n'est pas celle du
modèle fp32 : la calibration doit être refaite sur lui avant qu'il soit servi,
sans quoi le rang renvoyé serait faux. `torch.ao.quantization` est annoncé
comme retiré au profit de `torchao` ; il fonctionne encore avec torch 2.12.1,
la version de l'image, qui reste figée.

---

## 16 novembre 2026 — Le service tient son contrat, et l'API ne coûte presque rien

**Mesure.** Le service est lancé sur Kaggle, sur processeur (4 cœurs), avec le
mélange 3 à la révision `96a5860`, et appelé en HTTP. Les phrases sont tirées
comme dans `evaluation/mesures.py` : 200 phrases du jeu de test, graine 42.

| | phrase seule, médiane | p90 | lot de 16, médiane | p90 | par phrase en lot |
|---|---:|---:|---:|---:|---:|
| premier modèle, sans API (cahier des charges) | 0,866 s | 1,863 s | 15,194 s | 19,962 s | 0,835 s |
| mélange 3, à travers l'API | 0,934 s | 2,321 s | 12,605 s | 23,162 s | 0,788 s |

**Lecture.** L'écart de médiane, 0,07 s, réunit le coût de l'API et celui du
changement de modèle : il borne donc le premier, négligeable devant le
décodage. Le p90 des lots repose sur huit lots et reste indicatif. Sur
processeur, le lot demeure un faible levier, 0,788 s par phrase contre 0,934
seule, ce que le cahier des charges avait établi.

Contre le vrai modèle, le contrat est tenu : sorties complètes, ordre du lot
conservé, même sortie et même confiance pour une phrase seule ou dans un lot
(0,9968 dans les deux cas), refus à 422 d'une entrée vide ou trop longue,
Swagger servi. La limite de 191 octets ne refuse aucune des 850 phrases du jeu
de test.

**Ce que ça entraîne.** Sur processeur, environ une seconde par phrase, et
2,3 s pour une sur dix : c'est suffisant pour l'interface d'annotation, qui
présente les phrases une à une. Un lot de 64, le maximum accepté, prendrait
de l'ordre de cinquante secondes : un usage par lots soutenu demande un GPU.

---

## 15 novembre 2026 — Le modèle publié se charge avec Transformers 5, que le service fige

**Constat.** Au chargement, Transformers signale que la configuration demande de
lier la tête de sortie aux plongements, que les deux matrices diffèrent dans le
fichier, et qu'il ne les lie donc pas. Le `config.json` publié porte
`tie_word_embeddings: true`, à la racine comme dans `modèles_S/`, alors que
`google/byt5-small` porte `false`.

**Mesure.** Rechargé avec `false`, le modèle perd aussi le lien entre les
plongements de l'encodeur et du décodeur et la matrice partagée, que la
sauvegarde n'a écrite qu'une fois. Transformers 5 les signale manquants et les
réinitialise au hasard : les cinq phrases d'essai sortent vides. Avec `true`,
les sorties sont celles de la fiche du modèle, et les poids relus sont
identiques aux poids envoyés.

**Décision.** Le `config.json` publié n'est pas modifié : pour ce fichier et
Transformers 5, `true` est la valeur juste. L'entraînement et le service figent
`transformers==5.20.2`, version de Kaggle avec laquelle le modèle a été
entraîné et vérifié. Une version 4.x n'a pas été essayée ; elle lierait
vraisemblablement la tête de sortie aux plongements, et n'est pas prise en
charge.

**Ce que ça ne change pas.** Les confiances mesurées jusqu'ici en chargeant le
modèle depuis Hugging Face l'ont été avec le chargement juste.

---

## 15 novembre 2026 — Le troisième mélange devient le meilleur modèle

**Décision.** Le mélange 3 — 42 % de nilechat, corpus d'origine vu trois fois,
tirage par longueur, 15 000 pas — remplace le modèle de référence.

| sur 678 phrases | CER | exactitude | phrases avec latin |
|---|---:|---:|---:|
| baseline déterministe | 0,1699 | 17,6 % | 13 |
| référence | 0,1087 | 29,4 % | 4 |
| mélange 1 : 50 %, 1,5× | **0,1016** | 29,6 % | 7 |
| mélange 2 : 33 %, 3× | 0,1089 | 31,4 % | 2 |
| **mélange 3 : 42 %, 3×** | 0,1033 | **32,2 %** | **1** |

Meilleure exactitude mesurée, latin réduit à une phrase — la référence du jeu de
test en porte une aussi —, et un CER à 0,0017 du meilleur, soit −5 % par
rapport à la référence. Le mélange 1 garde le meilleur CER mais laisse du
français en latin dans sept phrases.

**Pourquoi ce résultat est plus solide que les précédents.** Le mélange 3 ne
diffère du mélange 2 que par la part de nilechat : même exposition au corpus
d'origine, même tirage par longueur. Le facteur confondant signalé dans
l'entrée précédente n'existe pas entre ces deux-là.

| catégorie | mélange 2 | mélange 3 | écart |
|---|---:|---:|---:|
| voyelles longues | 760 | 722 | −38 |
| espaces et agglutination | 274 | 254 | −20 |
| hamzas | 126 | 113 | −13 |
| latin résiduel | 18 | 13 | −5 |

Monter la part de nilechat de 33 à 42 % fait revenir les voyelles longues, et
le latin reste au plancher. Le modèle à deux variables de l'entrée précédente
se confirme, sans biais cette fois.

Les 38 opérations récupérées représentent 42 % de ce que le mélange 2 avait
perdu. Aucune courbe n'est tirée de là : le point à 50 % vient du mélange 1,
qui diffère aussi par le tirage et par l'exposition au corpus d'origine.

La perte de validation baisse à chaque contrôle, 0,1372 puis 0,1203 puis
0,1155, la plus basse des quatre runs. Aucun n'a convergé.

**Suite.** Le point 50/50 avec trois passages, qui devrait récupérer le reste
des voyelles sans perdre le latin. Il demande 17 520 pas, soit 12 h 40, au-delà
d'une session : il faudra une reprise sur point de sauvegarde, et le quota
hebdomadaire de la semaine en cours ne suffit plus.

---

## 14 novembre 2026 — Deux variables, deux effets, observés dans les deux sens

**Ce qui a été fait.** Un second mélange, conçu pour corriger la régression du
premier : le corpus d'origine rendu à ses **trois passages**, avec les mêmes
280 320 lignes de nilechat. Il faut pour cela 13 140 pas, qui ne tenaient pas
dans une session au rythme du premier mélange, 4,31 s par pas.

Le tirage des lots par longueur, écrit pour l'occasion, a ramené le pas à
**2,55 s**. Mesuré avant de s'engager, par un essai de 150 pas : la section
précédente avait vu plusieurs estimations de vitesse démenties. Il a fallu
l'écrire plutôt que l'emprunter, l'argument `group_by_length` ayant disparu de
la version de `transformers` installée sur Kaggle.

**Le résultat, sur les 678 phrases d'évaluation.**

| | CER | exactitude |
|---|---:|---:|
| référence | 0,1087 | 29,4 % |
| mélange 1 : 50 % de nilechat, origine 1,5× | **0,1016** | 29,6 % |
| mélange 2 : 33 % de nilechat, origine 3× | 0,1089 | **31,4 %** |

Les deux métriques imposées divergent : le second mélange a la meilleure
exactitude jamais mesurée, et un CER revenu au niveau de la référence.

**Ce qui explique la divergence.** Les deux runs, lus ensemble, se comportent
comme une expérience à deux variables.

| | part de nilechat | origine | voyelles longues | latin résiduel |
|---|---:|---:|---:|---:|
| mélange 1, contre la référence | 50 % | 1,5× | **−90** | +40 |
| mélange 2, contre le mélange 1 | 33 % | 3× | +91 | **−53** |

**Chaque variable déplace exactement la catégorie qu'on lui attribuait, et
dans les deux sens.** La part de nilechat commande la cohérence des voyelles
longues : ses cibles, produites par un même modèle, sont écrites de façon
homogène. L'exposition au corpus d'origine commande la translittération du
français : ce sont ses phrases qui l'enseignent. Le second mélange laisse du
français en latin dans 2 phrases, contre 7 au premier et 4 à la référence.

C'est ce qui distingue ce résultat d'une corrélation : l'effet apparaît quand
on monte la variable, et disparaît quand on la baisse.

**Ce qui a été écarté en chemin.** Les phrases longues se dégradent nettement
(CER de 0,0737 à 0,1164 sur ce thème). Trois sorties y montraient un arrêt
prématuré, une boucle et une transcription lettre à lettre, ce qui suggérait
un modèle instable. **Le comptage systématique l'a démenti** : une boucle par
modèle sur les trois, une seule troncature. La dégradation est diffuse, et
s'explique plus simplement par la baisse de la part de nilechat, faite de
phrases longues.

La sélection prématurée d'un point de sauvegarde est écartée aussi : la perte
de validation baisse à chaque contrôle, 0,1399, 0,1234 puis 0,1173, et c'est le
dernier point qui a été retenu.

Enfin, trois tranches affichaient un CER identique au dix-millième d'un run à
l'autre. Vérifié : 516 prédictions sur 850 sont communes aux deux modèles, ce
qui rend ces égalités plausibles. Ce n'était pas une erreur de copie.

**Facteur confondant.** Le second mélange diffère du premier par le tirage par
longueur en plus de la proportion. La lecture ci-dessus repose sur le fait que
chaque catégorie a bougé dans le sens prédit par la proportion ; elle ne
démontre pas que le tirage est neutre. Les runs suivants tirant tous par
longueur, ils se compareront entre eux sans ce biais.

**Suite.** Viser les deux effets à la fois : origine 3× et forte part de
nilechat. Le point exact, 50/50, demande 17 520 pas, soit 12 h 40, hors
session. 15 000 pas donnent 42 % de nilechat en 10 h 50.

---

## 14 novembre 2026 — Le mélange avec nilechat améliore le CER de 6,5 %

**Résultat.** Premier gain mesuré depuis l'entraînement initial, sur les 678
phrases d'évaluation.

| modèle | CER | exactitude |
|---|---:|---:|
| ByT5, corpus d'origine | 0,1087 | 29,4 % |
| ByT5, augmentation de graphies | 0,1085 | 29,5 % |
| **ByT5, mélange avec nilechat** | **0,1016** | **29,6 %** |

**Protocole.** Budget identique à la référence : 8 760 pas de 64. Mélange
50/50, si bien que le corpus d'origine n'est vu que **1,5 fois au lieu de 3,0**,
et que 8,9 % du candidat seulement est montré. C'était délibéré : un gain obtenu
avec moins d'exposition aux bonnes données ne peut venir que du candidat.

Le lot par carte est passé de 16 à 8, avec 4 pas d'accumulation au lieu de 2.
Les phrases longues de nilechat — 91 octets en médiane contre 21 pour
`doda_synthetic` — saturaient la mémoire au premier pas. Le lot effectif reste
64, ce que le script imprime désormais.

**Le corpus.** `UBC-NLP/nilechat-arabizi-mor` n'a pas de colonne arabe ; il a
été joint à sa source `nilechat-lhv-mor` sur les identifiants. L'autre source,
`fw-edu`, a été écartée : texte web éducatif massivement en arabe standard, où
le modèle de translittération avait parfois **traduit** vers le darija au lieu
de translittérer. 3 282 816 paires au final, alignées à 99,8 % au niveau de la
phrase. **Leur arabizi est produit par un modèle de langue**, et ne doit jamais
être présenté comme écrit par des humains.

Mesuré avant tout entraînement : **40,7 %** du vocabulaire absent de notre
corpus y figure, quand toutes nos sources sont à 0 %. La distance orthographique
au jeu de test, 1,507, est à 0,790 de la ponctuation ; hors ponctuation, l'écart
côté cible n'est que de 0,071.

Au mélange, la purge a retiré **628 fuites vers le jeu de test** et 134 029
doublons internes, avec des chiffres identiques d'un run à l'autre.

**Licence.** La source arabe est en CC-BY-ND, qui interdit de distribuer une
œuvre dérivée. Le fichier de paires ne se publie pas ; seul le script
d'extraction l'est.

**Où est le gain.** Voyelles longues, −90 opérations, soit **78 % du gain**. La
catégorie que l'augmentation de graphies n'avait pas su bouger. Ponctuation et
espaces, −30 chacune.

**Ce qui ne s'est pas passé comme prévu.** Le vocabulaire manquant ne progresse
que de 23,0 à 26,1 % de mots retrouvés, contre 73,4 à 74,6 % pour les mots déjà
connus : le gain porte sur l'écriture de mots connus, rendue plus cohérente, et
non sur du vocabulaire neuf. Le modèle n'ayant vu que 8,9 % du candidat, la
couverture mesurée sur l'ensemble n'a pas pu agir.

Deux prédictions démenties au passage. La ponctuation devait se dégrader à
cause des virgules de nilechat, présentes dans 79,3 % de ses phrases ; elle
s'améliore. Et le gain devait se concentrer sur les formes inconnues.

**Une régression réelle.** Le modèle laisse davantage de français en latin —
`kan9ra l'informatique f la fac` rendu `كنقرا l'informatique ف la fac` — soit 7
phrases au lieu de 4 et +40 opérations. Hypothèse non vérifiée : moins
d'exposition au corpus d'origine, qui porte les exemples de translittération du
français.

**Une régression apparente.** Tetouan affiche +25 % de CER. C'est **7
opérations sur 48 phrases**, portées par trois mots isolés : du bruit, pas un
résultat.

**Suite.** Rendre au corpus d'origine ses trois passages et montrer davantage
de nilechat — les deux problèmes ont ce même remède. L'obstacle est le temps :
4,31 s par pas avec le lot réduit, et ce run a duré environ 10 h 30 pour une
limite de session à 12 h.

---

## 14 novembre 2026 — Le profil orthographique est redéfini

**Correction.** La mesure de distance orthographique au jeu de test a été
réécrite dans `profil_corpus.py`, sur dix traits nommés, et recalculée pour
toutes les sources. Son classement **contredit** une mesure antérieure qui
plaçait `daa-pairs` comme la source la plus proche du jeu de test : elle en est
ici l'une des plus lointaines, à 1,087, et `doda_synthetic` la plus proche, à
0,229.

L'ancienne mesure n'avait pas été écrite et ne peut être reproduite. Les deux
séries ne doivent pas être citées ensemble ; seule la nouvelle fait foi.

---

## 12 novembre 2026 — Le réordonnancement des hypothèses est écarté

**Décision.** Le décodage glouton reste la méthode de production. Quatre
sélections ont été essayées, toutes moins bonnes.

| sélection | CER | exactitude |
|---|---:|---:|
| **décodage glouton** | **0,1087** | **29,4 %** |
| première hypothèse du faisceau | 0,1158 | 30,5 % |
| consensus, sans réglage | 0,1143 | 25,5 % |
| mélange réglé sur le dev | 0,1138 | 28,8 % |
| mélange réglé sur le test | 0,1163 | 26,7 % |
| choix parfait (plafond) | 0,0622 | 54,3 % |

**Ce que ça établit.** L'information *est* dans le faisceau : la référence exacte
y figure pour 54,3 % des phrases, et le modèle n'en choisit que 30,5 %. Mais
aucun signal non supervisé ne la distingue — ni la vraisemblance du modèle, ni
l'accord entre hypothèses, ni la fluidité en bigrammes.

Le premier échec avait été attribué à la distribution du jeu de réglage. Cette
explication est **réfutée** : régler sur un prélèvement du jeu de test donne
0,1163, soit pire que régler sur le développement. Le CER de réglage, 0,1149,
est proche de celui d'évaluation, ce qui écarte aussi le surajustement.

**Condition de réouverture.** Un réordonnanceur appris sur des exemples
étiquetés de la bonne distribution. Il en existe 172, ce qui est trop peu. La
piste est donc bloquée sur la campagne d'annotation, et avec un plafond à 24
points d'exactitude elle mérite d'être rouverte dès que possible.

---

## 12 novembre 2026 — Un jeu de réglage est prélevé sur TerjamaBench

**Décision.** 20 % du jeu de test, soit 172 phrases, deviennent un jeu de
réglage. Les 678 restantes sont la seule base d'évaluation.

**Motif mesuré.** Le jeu de développement issu de `make_splits.py` vient des
sources d'entraînement, et le modèle y atteint **0,0473 de CER contre 0,1146 sur
le test**. Toute décision de sélection réglée dessus l'était sur une
distribution où le problème n'existe pas.

**Pourquoi 20 %.** À 50 %, Nador tomberait à 11 phrases et Guelmim à 13, et
l'analyse par variété régionale — qui a produit le principal résultat du projet
— deviendrait du bruit. À 20 % elles gardent 17 et 20 phrases. Le prélèvement
est stratifié sur le couple thème et variété : chaque strate ressort entre
19,0 % et 22,2 %.

**Conséquence.** Les chiffres publiés avant cette date portaient sur 850
phrases et ne sont pas comparables.

| | avant, sur 850 | **désormais, sur 678** |
|---|---:|---:|
| baseline | 0,1735 / 18,2 % | **0,1699 / 17,6 %** |
| ByT5 | 0,1093 / 29,2 % | **0,1087 / 29,4 %** |

---

## 12 novembre 2026 — La table de normalisation externe n'est pas retenue

**Décision.** La table phonétique vers conventionnel extraite de
`atlasia/DODa-audio-dataset` n'est pas appliquée à la métrique principale. Elle
est publiée comme mesure secondaire.

**Mesure.** 1 999 substitutions attestées, tirées de 12 052 phrases que les mêmes
annotateurs ont écrites sous deux orthographes. Appliquée des deux côtés : +3,5
points d'exactitude au modèle, +4,6 à la baseline, pour un CER dégradé de
0,1087 à 0,1105.

**Motifs.** Elle améliore une métrique imposée et dégrade l'autre, ce qui rend
l'arbitrage indéfendable. Elle est tirée d'un corpus et appliquée à un autre.
Et elle n'est pas nécessaire pour conclure, l'écart entre modèle et baseline se
conservant dans les deux régimes.

**Ce qu'elle apporte quand même.** Elle établit que 3,5 points de ce qu'on
compte comme erreurs sont des variantes que des annotateurs indépendants
tiennent pour équivalentes. C'est un plafond de convention chiffré de
l'extérieur.

---

## 12 novembre 2026 — DODa-audio remplace sa variante V3

**Décision.** `donnees/extraction/extract_doda_audio.py` tire
`atlasia/DODa-audio-dataset` à la place du CSV de la V3 récupéré à la main.

**Vérification.** Les 8 697 paires conservées de la V3 s'y retrouvent toutes,
aucune perte, et **3 668 s'y ajoutent**. Licence MIT.

*Corrigé le 21 novembre : ce chiffre est faux, 360 paires seulement sont
nouvelles pour le corpus. Voir l'entrée de cette date.*

**Effet de bord.** C'était la dernière source sans script. Le pipeline est
intégralement reproductible, ce qu'exige ENF-04.

---

## 12 novembre 2026 — L'augmentation de graphies ne change rien

**Résultat nul**, et il faut l'écrire ainsi plutôt que négatif.

**Protocole.** Corpus porté de 186 861 à 498 290 lignes par permutation de
graphies arabizi attestées, la cible n'étant jamais touchée. Entraînement à
**budget de calcul constant** : 8 760 pas dans les deux cas, donc chaque phrase
arabe vue exactement 3,0 fois, la seule variable changée étant le nombre de
graphies par phrase.

**Mesure.** CER 0,1085 contre 0,1087 pour la référence, et la catégorie visée
n'a pas bougé. **Le classement entre les deux runs s'inverse selon qu'on mesure
sur 850 ou sur 678 phrases** : l'écart est du bruit.

**Ce que ça élimine.** Varier la graphie de l'entrée n'apprend rien au modèle.
L'hypothèse de départ — « les données ne contiennent aucune variation » — était
d'ailleurs fausse : au niveau du mot, un mot fréquent s'écrit de 2 façons en
médiane.

---

## 12 novembre 2026 — Le budget d'erreur est décomposé au niveau du mot

**Ce que ça corrige.** La ventilation par caractère attribuait 43,5 % des
erreurs aux voyelles longues. Au niveau du mot, elles n'en pèsent que **14,1 %**
du coût : la plupart de ces caractères sont des débris à l'intérieur de mots
entièrement faux.

| | part du coût |
|---|---:|
| mot différent | 47,9 % |
| découpage différent | 38,0 % |
| variante de voyelle longue | 14,1 % |

64,6 % des mots sont déjà exactement justes.

**Deux plafonds de convention mesurés.** Canoniser les variantes orthographiques
gagne 2,1 points d'exactitude ; ignorer les espaces en gagne 3,9, 33 phrases sur
850 ne différant de la référence que par leurs frontières de mots. Les deux
laissent le CER inchangé.

**Ce qui reste.** 22,9 % des mots faux sont des formes **jamais vues** dans le
corpus, concentrées sur Guelmim (158,7 ‰) et Tetouan (108,4 ‰) contre 19,1 ‰ à
Marrakech. C'est une lacune de couverture lexicale, pas de répartition : on ne
suréchantillonne pas des mots absents.

---

## 5 novembre 2026 — Premier entraînement de ByT5

**Résultat.** CER 0,1087 et 29,4 % d'exactitude contre 0,1699 et 17,6 % pour la
baseline déterministe entraînée sur les mêmes données. 8 760 pas, 7 h 38, deux
cartes T4.

**Défaut principal, trouvé en lisant les erreurs une par une.** Le modèle
remplace systématiquement la forme régionale par la forme majoritaire :
`bit` devient بغيت au lieu de بيت, le futur tétouanais `عـ` devient le `غـ`
casablancais, `شني` devient `شنو`. Il normalise le dialecte par-dessus la
translittération, ce qu'aucune exigence ne demande.

Sur les 21 phrases de Nador, le nombre d'opérations d'édition reste de 47 pour
la baseline comme pour le modèle. Sept heures et demie d'entraînement n'y ont
corrigé aucun caractère.

---

## Ce qui est au cahier des charges et n'est pas repris ici

Les arbitrages PO-01 à PO-06 — convention orthographique de sortie, alternance
codique française, éléments non lexicaux, stratégie de corpus, score de
confiance, taille du modèle et décodage — ainsi que l'état du jeu de test et
l'écart assumé sur son registre.
