# Journal des décisions

Le cahier des charges fixe les décisions d'arbitrage PO-01 à PO-06 et n'est plus
modifié. Ce journal consigne ce que la mesure a établi depuis, y compris les
résultats négatifs — une spécification dit ce qu'on fait, pas ce qu'on a essayé
sans succès, et c'est pourtant ce second registre qui explique le premier.

Chaque entrée dit ce qui a été décidé, sur quelle mesure, et ce que ça entraîne.
Les plus récentes d'abord.

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
