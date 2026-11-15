# Protocole d'annotation

Ce document s'adresse aux personnes qui vont annoter. Il tient en dix minutes
de lecture, et il faut l'avoir lu en entier avant de commencer : la moitié des
désaccords entre annotateurs vient de règles qu'on croyait évidentes.

## Ce qu'on vous demande

Vous recevez des phrases en **arabizi** — du darija écrit en lettres latines,
avec des chiffres pour les sons qui n'existent pas en français.

```
chrit tomobila jdida w mchit biha l dar
```

Vous écrivez la même phrase en **lettres arabes**.

```
شريت طوموبيلا جديدة ومشيت بيها لدار
```

C'est tout. Vous ne traduisez pas, vous ne corrigez pas, vous ne reformulez
pas. Vous changez l'alphabet, rien d'autre.

## La règle la plus importante

**Écrivez ce que la personne a écrit, pas ce qu'elle aurait dit à Casablanca.**

Notre modèle a un défaut qu'on a mesuré : il remplace les formes régionales par
les formes de Casablanca. Il écrit `بغيت` quand quelqu'un a écrit `bit`, et
`غاتجيك` quand un Tétouanais a écrit `3atjik`. Si vous faites la même chose, le
jeu de test ne pourra plus détecter ce défaut, et tout ce travail n'aura servi
à rien.

| arabizi | à écrire | **pas** |
|---|---|---|
| `bit nmchi` | بيت نمشي | ~~بغيت نمشي~~ |
| `3atjik kbira` | عتجيك كبيرة | ~~غاتجيك كبيرة~~ |
| `chni 3ndk` | شني عندك | ~~شنو عندك~~ |

Si une forme vous paraît bizarre ou « pas du vrai darija », c'est probablement
qu'elle vient d'une autre région que la vôtre. Gardez-la.

## Les règles d'écriture

**Pas de vocalisation.** Aucune fatha, kasra, damma ni chadda.

> `bzzaf` → بزاف &nbsp;&nbsp; et non ~~بزّاف~~

**Ponctuation en arabe.** Le point d'interrogation s'écrit `؟` et la virgule
`،`. Le point et le point d'exclamation ne changent pas.

> `fin ghadi?` → فين غادي؟

**Les chiffres restent des chiffres.** Quand un chiffre est un nombre, il reste
tel quel. Quand il remplace une lettre, il devient cette lettre.

> `chrit 3 tomobilat` → شريت 3 طوموبيلات
>
> Ici le `3` de `tomobilat` n'existe pas, mais dans `3afak` le `3` est la
> lettre ع : عافاك.

**Les émojis restent où ils sont**, sans rien changer.

**Les rires et les allongements se gardent.** `hhhhh` reste هههه, `zwiiiin`
reste زويين. N'abrégez pas.

## Le français au milieu de la phrase

Les Marocains mélangent le français au darija. **Le mot français s'écrit en
lettres arabes, selon sa prononciation.**

| arabizi | à écrire |
|---|---|
| `mchit la pharmacie` | مشيت للفرماسي |
| `khdit rendez-vous` | خديت رونديڤو |
| `chrit fauteuil jdid` | شريت فوتاي جديد |

Pour les mots étrangers récents, il n'y a pas d'orthographe officielle.
`gaming` peut s'écrire الغيمينج ou الڭامينغ, et les deux sont acceptables.
**Écrivez comme ça se prononce, et ne revenez pas dessus.** On a mesuré que ces
cas font du désaccord entre annotateurs sans que personne ait tort.

Les sigles et noms propres étrangers qui s'écrivent en lettres latines dans la
vie courante restent en lettres latines : `ENCG` reste ENCG.

## D'où viennent les phrases

**De vraies conversations**, pas de phrases inventées. C'est la raison d'être
de cette campagne : notre jeu de test actuel est écrit par des annotateurs à qui
on a demandé de produire de l'arabizi, et ça ne ressemble pas à ce que les gens
écrivent vraiment sur WhatsApp.

**On cherche en priorité des locuteurs du Rif, du Souss, de Tétouan et du Sud.**
Ce sont les régions où le modèle échoue le plus, et celles qui manquent le plus
dans nos données.

### Consentement et anonymat, non négociables

Avant de fournir des messages, la personne doit savoir et accepter que :

- ses phrases serviront à un projet d'école publié en accès libre ;
- elles seront publiées, donc lisibles par n'importe qui ;
- elle peut demander le retrait d'une phrase à tout moment.

Avant d'intégrer une phrase, **retirez systématiquement** : les prénoms et noms,
les numéros de téléphone, les adresses, les noms d'établissement ou d'entreprise
qui identifient quelqu'un, et tout ce qui touche à la santé, à l'argent ou à la
vie privée d'un tiers.

Dans le doute, écartez la phrase. Une phrase en moins ne coûte rien ; une
donnée personnelle publiée ne se rattrape pas.

## La double annotation

**Une phrase sur dix est annotée deux fois, par deux personnes différentes qui
ne voient pas le travail l'une de l'autre.** Ce n'est pas un contrôle de votre
sérieux : c'est la mesure qui dit si nos règles d'écriture sont assez précises.

Si deux personnes compétentes écrivent la même phrase différemment, le problème
est dans ce document, pas dans leur travail. On regarde alors **où** porte le
désaccord — voyelles longues, espaces, hamzas — et on ajoute la règle qui
manque.

C'est aussi pour ça qu'il ne faut pas se concerter : deux annotations faites
ensemble ne mesurent plus rien.

### Comment l'accord est mesuré

Sur les phrases annotées deux fois, on calcule le taux de caractères différents
entre les deux versions, et la part de phrases écrites exactement pareil. La
ventilation par catégorie est produite par `erreurs.py`, le même outil qui sert
à analyser les erreurs du modèle.

**Objectif provisoire : moins de 5 % de caractères divergents, et au moins 60 %
de phrases identiques.** Ces seuils sont provisoires parce que personne ne sait
ce qui est atteignable sur cette tâche. Les trente premières phrases en double
servent à les fixer pour de bon.

Si le désaccord se concentre sur une seule catégorie, la règle correspondante
est ajoutée ici et le lot est réannoté. Si le désaccord est dispersé, c'est que
la tâche est intrinsèquement ambiguë et le seuil est revu, avec la mesure à
l'appui.

## Format

Un fichier CSV par annotateur, en UTF-8, avec exactement ces colonnes :

| colonne | contenu |
|---|---|
| `id` | identifiant de la phrase, fourni avec le lot |
| `transliteration` | l'arabizi, recopié sans y toucher |
| `darija` | votre transcription en lettres arabes |
| `annotateur` | votre identifiant |
| `region` | région du locuteur d'origine, si connue |
| `remarque` | vide, ou ce qui vous a fait hésiter |

La colonne `remarque` compte autant que les autres. Une hésitation notée, c'est
une règle manquante repérée avant qu'elle ne fausse cinquante phrases.

## Ce que ce jeu de données deviendra

Il sert à deux choses, et il ne faut jamais les mélanger.

Une partie **mesure** le modèle, et ne doit jamais servir à le régler, sous
peine de ne plus rien mesurer du tout.

L'autre sert à **régler** les décisions de sélection, à la place du jeu de
développement actuel, qui vient des mêmes sources que l'entraînement et sur
lequel le modèle réussit deux fois et demie mieux que dans la réalité.

Le découpage entre les deux est fait une fois, avant la première évaluation, et
n'est plus touché.
