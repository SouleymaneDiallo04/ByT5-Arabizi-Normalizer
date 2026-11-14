# Journal des décisions

Le cahier des charges fixe les décisions d'arbitrage PO-01 à PO-06 et n'est plus
modifié. Ce journal consigne ce que la mesure a établi depuis, y compris les
résultats négatifs — une spécification dit ce qu'on fait, pas ce qu'on a essayé
sans succès, et c'est pourtant ce second registre qui explique le premier.

Chaque entrée dit ce qui a été décidé, sur quelle mesure, et ce que ça entraîne.
Les plus récentes d'abord.

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

**Décision.** `Extractor/extract_doda_audio.py` tire
`atlasia/DODa-audio-dataset` à la place du CSV de la V3 récupéré à la main.

**Vérification.** Les 8 697 paires conservées de la V3 s'y retrouvent toutes,
aucune perte, et **3 668 s'y ajoutent**. Licence MIT.

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
