# TP INF232 — Statistiques et Analyse de Données

**Groupe 36 — Chef de groupe : Puyuenpouopawouo Mounchili Adrien**
**Thème retenu : THÈME A — Service de santé universitaire**

---

## 1. Choix du thème

Le groupe a retenu le **Thème A : Service de santé universitaire**. Le service
de santé de l'Université de MBOUDA a mené un dépistage de routine sur un
échantillon d'étudiants, à partir duquel il souhaite obtenir des réponses
concrètes en statistique descriptive et en classification.

Langage retenu : **Python (pandas, numpy, scikit-learn, matplotlib, seaborn,
scipy)**, choisi car son écosystème réunit dans un même langage la manipulation
de données, les tests statistiques et les algorithmes de classification
supervisée/non supervisée nécessaires aux quatre questions du thème.

---

## 2. Mécanisme de génération des données personnalisées (Annexe)

### 2.1 Principe de l'algorithme

Le nom complet du chef de groupe est d'abord normalisé : mis en majuscules,
débarrassé de ses accents et de ses espaces.

> `"Puyuenpouopawouo Mounchili Adrien"` → `"PUYUENPOUOPAWOUOMOUNCHILIADRIEN"`

Cette chaîne est ensuite transformée en un entier reproductible (la **graine
du groupe**) via une fonction de hachage polynomial déterministe :

```
seed = ( Σ ord(caractère_i) × P^i )  mod M
```

avec `P = 31` (nombre premier utilisé comme base du hachage polynomial) et
`M = 2^31 - 1` (nombre premier de Mersenne, choisi comme modulo pour obtenir
un entier de taille raisonnable). Ce choix garantit :

- **le déterminisme total** : relancer le programme avec le même nom donne
  toujours exactement la même graine ;
- **une très faible probabilité de collision** entre deux noms différents,
  chaque caractère contribuant avec un poids différent (`P^i`) selon sa
  position dans la chaîne ;
- **un entier directement utilisable** comme graine d'un générateur
  pseudo-aléatoire standard.

Cette graine initialise ensuite `numpy.random.default_rng(seed)`, qui produit
l'ensemble des données du groupe : IMC, tension artérielle systolique (TAS),
une variable latente interne (non transmise), et l'étiquette de risque.

Les deux variables physiologiques mesurées, choisies et justifiées dans le
contexte étudiant, sont :

- **IMC (Indice de Masse Corporelle, kg/m²)** — mesure physiologique standard
  de dépistage, facilement interprétable par une équipe médicale ;
- **TAS (Tension Artérielle Systolique, mmHg)** — seconde mesure physiologique
  de routine, physiologiquement liée à l'IMC.

L'étiquette **Risque** (`a_risque` / `non_a_risque`) n'est pas une simple règle
de seuil sur IMC et TAS : elle est générée par un modèle logistique latent qui
inclut aussi une **variable cachée non transmise** (prédisposition
individuelle simulée : hérédité, mode de vie, etc.) :

```
z = b0 + b1·IMC_std + b2·TAS_std + b3·L
p = 1 / (1 + e^(-z))
Risque ~ Bernoulli(p)
```

Conséquence volontaire de ce choix : plus le profil d'un étudiant est
"limite" (p proche de 0.5), plus l'étiquette dépend d'une part de hasard
irréductible (la variable latente inconnue). Plus le profil est "extrême",
plus l'étiquette est quasi certaine. Cette propriété **n'est pas un artefact
mais un choix de conception assumé**, qui permet de répondre honnêtement, en
Q2 et Q4, aux questions du directeur sur la fiabilité et les limites d'une
estimation ou d'une prédiction automatique — un service de santé réel ne
dispose jamais de toute l'information nécessaire à un risque parfaitement
prévisible.

### 2.2 Taille de l'échantillon

**N = 320 étudiants.** Justification : les méthodes mobilisées (régression
linéaire, K-means avec recherche du k optimal par silhouette, classification
supervisée avec séparation train/test 80/20) nécessitent un nombre
d'individus suffisant pour rester stables et interprétables. Une règle
empirique courante demande au moins une trentaine d'individus par
groupe/classe attendu ; avec 2 classes de risque et plusieurs profils
potentiels envisagés, un minimum de 120 à 150 individus suffirait déjà. 320
offre une marge confortable, notamment pour que l'échantillon de test de la
Q4 (20 %, soit 64 étudiants) reste lui-même statistiquement exploitable.

### 2.3 Code de génération

```python
import unicodedata
import numpy as np
import pandas as pd

def normaliser_nom(nom_complet: str) -> str:
    sans_accents = unicodedata.normalize("NFKD", nom_complet)
    sans_accents = "".join(c for c in sans_accents if not unicodedata.combining(c))
    return "".join(sans_accents.upper().split())

def nom_vers_graine(nom_complet: str) -> int:
    chaine = normaliser_nom(nom_complet)
    P = 31
    M = (2 ** 31) - 1
    graine = 0
    for i, char in enumerate(chaine):
        graine = (graine + ord(char) * pow(P, i, M)) % M
    return graine

def generer_donnees(graine: int, n: int = 320):
    rng = np.random.default_rng(graine)

    imc = rng.normal(loc=22.5, scale=3.4, size=n)
    imc = np.clip(imc, 15.5, 36.0)

    tas = 100 + 1.15 * (imc - 22.5) + rng.normal(loc=0, scale=8.5, size=n)
    tas = np.clip(tas, 92.0, 158.0)

    latente = rng.normal(loc=0.0, scale=1.0, size=n)

    imc_std = (imc - imc.mean()) / imc.std()
    tas_std = (tas - tas.mean()) / tas.std()

    b0, b1, b2, b3 = -0.4, 0.9, 0.75, 0.9
    z = b0 + b1 * imc_std + b2 * tas_std + b3 * latente
    p = 1 / (1 + np.exp(-z))
    risque_bin = rng.binomial(1, p)

    df = pd.DataFrame({
        "id_etudiant": [f"ETU{i+1:04d}" for i in range(n)],
        "IMC": np.round(imc, 2),
        "TAS": np.round(tas, 1),
        "Risque": np.where(risque_bin == 1, "a_risque", "non_a_risque"),
    })
    return df
```

*(Code complet, avec docstrings détaillées, disponible dans
`Application/data_generator.py`.)*

### 2.4 Graine obtenue et preuve de reproductibilité

| Élément | Valeur |
|---|---|
| Nom du chef de groupe | Puyuenpouopawouo Mounchili Adrien |
| Chaîne normalisée | `PUYUENPOUOPAWOUOMOUNCHILIADRIEN` |
| **Graine obtenue** | **162427331** |
| Nombre d'individus générés | 320 |

Le script `data_generator.py` a été exécuté deux fois de suite de manière
indépendante : les deux exécutions produisent **rigoureusement la même
graine (162427331)** et les mêmes 320 lignes de données, confirmant le
déterminisme du mécanisme.

### 2.5 Extrait représentatif des données produites

| id_etudiant | IMC | TAS | Risque |
|---|---|---|---|
| ETU0001 | 25.64 | 94.6 | non_a_risque |
| ETU0002 | 24.02 | 106.3 | a_risque |
| ETU0003 | 24.03 | 101.1 | a_risque |
| ETU0004 | 30.42 | 112.9 | a_risque |
| ETU0005 | 20.59 | 97.4 | non_a_risque |

Répartition globale de l'étiquette sur les 320 étudiants : **184
non_a_risque / 136 a_risque**.

---

## 3. Question 1 — Distribution de l'IMC

**Question du directeur :** *« Comment se répartit le premier indicateur
physiologique ? Y a-t-il des étudiants dont la mesure sort vraiment du lot ?
Comment présenter cela simplement à mon équipe médicale ? »*

**Méthode mobilisée : statistique descriptive univariée.** Les indicateurs de
tendance centrale (moyenne, médiane) et de dispersion (écart-type, quartiles)
résument la distribution ; la méthode de l'écart interquartile (IQR) détecte
les valeurs atypiques de façon robuste et facilement explicable à une équipe
non statisticienne.

### Résultats

| Indicateur | Valeur |
|---|---|
| Effectif | 320 étudiants |
| Moyenne | 22.69 kg/m² |
| Médiane | 22.64 kg/m² |
| Écart-type | 3.25 kg/m² |
| Minimum / Maximum | 15.50 / 34.45 kg/m² |
| Q1 (25 %) / Q3 (75 %) | 20.48 / 25.10 kg/m² |
| Intervalle "normal" (méthode IQR) | [13.55 ; 32.02] |
| **Étudiants atypiques détectés** | **1** (ETU0144 = 34.45) |

![Distribution de l'IMC](figures/q1_distribution_imc.png)

### Réponse en langage simple pour le directeur

> La grande majorité des étudiants ont un IMC compris entre 13,6 et 32,0
> kg/m², la valeur typique se situant autour de 22,6. Un seul étudiant sort
> nettement de cette zone (IMC de 34,5) et mériterait une attention
> particulière lors du dépistage.

### Discussion des limites

L'échantillon est généré et non observé sur le terrain : la distribution
suppose une population étudiante "standard" et ne capture pas d'éventuelles
particularités locales (régime alimentaire, activité physique propre au
campus de MBOUDA). La méthode IQR, bien que robuste, reste une convention
statistique (facteur 1.5) : avec un seul cas détecté ici, la limite de la
méthode se voit bien — un étudiant juste en-dessous du seuil (par exemple à
IMC = 31) peut être tout aussi préoccupant cliniquement sans être signalé
statistiquement. Le signalement automatique ne remplace pas un avis médical
individuel.

---

## 4. Question 2 — Relation entre IMC et TAS

**Question du directeur :** *« Les deux indicateurs évoluent-ils ensemble ?
Pourrait-on n'en mesurer qu'un seul et estimer l'autre ? À partir de quand
une telle estimation deviendrait-elle dangereuse ? »*

**Méthode mobilisée : statistique bivariée.** Corrélation de Pearson pour
mesurer la force du lien linéaire, régression linéaire simple (TAS ~ IMC)
pour quantifier la relation et permettre une estimation, puis **analyse des
résidus** pour répondre précisément à la question du seuil de danger.

### Résultats

| Indicateur | Valeur |
|---|---|
| Corrélation de Pearson (IMC, TAS) | **r = 0.327** (p = 2.14 × 10⁻⁹, significatif) |
| Régression : TAS = a + b·IMC | b = 0.808, a = 83.08 |
| R² | 0.107 (10,7 % de la variance de la TAS expliquée par l'IMC seul) |
| Écart-type des résidus (zone centrale, IMC proche de la médiane) | 7.96 mmHg |
| Écart-type des résidus (zone extrême) | 6.60 mmHg |
| Zone d'estimation fiable | IMC entre ~19.4 et ~25.9 kg/m² |

![Relation IMC / TAS](figures/q2_regression_imc_tas.png)

### Réponse en langage simple pour le directeur

> Oui, les deux mesures évoluent globalement ensemble, mais le lien est
> plus modeste que ce qu'on pourrait espérer (corrélation r = 0,33) :
> l'IMC seul n'explique qu'environ 11 % de la variation de la tension. On
> peut estimer la TAS à partir de l'IMC pour les étudiants dont l'IMC est
> proche de la moyenne (19,4 à 25,9 kg/m² environ), avec une marge d'erreur
> qui reste raisonnable dans cette zone. Au-delà, la fiabilité de
> l'estimation reste comparable voire légèrement meilleure sur cet
> échantillon précis — mais la faiblesse générale du lien (R² = 0,11)
> justifie à elle seule la prudence : ce lien est trop ténu pour remplacer
> systématiquement une vraie mesure de tension par une simple estimation à
> partir de l'IMC.

### Discussion des limites

Un R² de seulement 0,107 signifie que **près de 90 % de la variabilité de
la TAS reste inexpliquée par l'IMC seul** : le lien, bien que statistiquement
significatif (p très petit, grâce à la taille de l'échantillon), reste faible
en pratique. Il est important ici de bien distinguer **significativité
statistique** (le lien n'est pas dû au hasard) et **utilité pratique** (le
lien est-il assez fort pour être exploité en routine) — ce n'est pas le cas
ici. La comparaison des résidus entre zone centrale et zone extrême ne suit
pas ici le sens intuitif attendu (résidus légèrement plus élevés au centre
que dans les extrêmes) : ceci illustre que sur un échantillon particulier,
le signal peut être discret, et qu'une conclusion définitive nécessiterait
un test statistique formel (ex. test de Breusch-Pagan) plutôt qu'une simple
comparaison de deux écarts-types.

---

## 5. Question 3 — Profils de santé naturels

**Question du directeur :** *« Existe-t-il des groupes naturels de profils de
santé parmi mes étudiants ? Combien, et comment les décrire simplement ? »*

**Méthode mobilisée : classification non supervisée (K-means).** Les deux
variables sont standardisées avant le clustering. Le nombre de clusters k est
choisi par la **méthode de la silhouette**, plus rigoureuse qu'un choix
arbitraire. Le clustering est effectué **uniquement sur IMC et TAS**, sans
utiliser l'étiquette Risque — c'est ce qui distingue une démarche non
supervisée d'une démarche supervisée.

### Résultats

| k | Score de silhouette |
|---|---|
| 2 | 0.382 |
| 3 | 0.333 |
| 4 | 0.374 |
| **5** | **0.387 (retenu)** |
| 6 | 0.376 |
| 7 | 0.359 |

| Profil | Effectif | IMC moyen | TAS moyenne | % déjà étiquetés "à risque" |
|---|---|---|---|---|
| Profil 0 | 66 | 18.5 kg/m² | 95.1 mmHg | 9 % |
| Profil 1 | 61 | 26.8 kg/m² | 105.1 mmHg | 79 % |
| Profil 2 | 84 | 21.5 kg/m² | 105.2 mmHg | 42 % |
| Profil 3 | 25 | 24.7 kg/m² | 119.2 mmHg | 84 % |
| Profil 4 | 84 | 23.6 kg/m² | 94.6 mmHg | 31 % |

![Clustering des profils](figures/q3_clustering_profils.png)

### Réponse en langage simple pour le directeur

> Sans utiliser votre classement "à risque / non à risque" existant, les
> données font naturellement apparaître **5 profils de santé distincts**
> parmi vos étudiants :
> - un profil "IMC bas, tension basse" peu concerné par le risque (9 %) ;
> - un profil "IMC élevé, tension modérée" très majoritairement à risque
>   (79 %) ;
> - un profil "IMC moyen, tension plutôt haute" partagé (42 % à risque) ;
> - un profil minoritaire mais très marqué, "tension nettement élevée",
>   presque systématiquement à risque (84 %) ;
> - un profil "IMC moyen, tension basse" peu concerné (31 %).
>
> Ces cinq profils ne recoupent pas parfaitement votre classement binaire
> actuel : chacun contient un mélange plus ou moins marqué d'étudiants à
> risque et non à risque, ce qui suggère qu'un classement en seulement deux
> catégories simplifie une réalité plus nuancée, organisée autour de
> plusieurs combinaisons typiques d'IMC et de tension.

### Discussion des limites

Le score de silhouette retenu (0.387 pour k=5) reste proche de celui obtenu
pour k=2 (0.382) ou k=4 (0.374) : le choix de 5 profils est défendable mais
pas radicalement supérieur aux alternatives, ce qui signifie que la
segmentation "correcte" du point de vue statistique n'est pas unique ni
définitive. Le profil 3, en particulier, ne regroupe que 25 étudiants
(moins de 8 % de l'échantillon) : un profil aussi minoritaire mérite d'être
interprété avec prudence, sa description pouvant être sensible à quelques
individus seulement. Comme pour toute analyse de clustering, le nombre de
profils "réel" dépend entièrement des deux variables choisies ; avec
d'autres mesures physiologiques, d'autres regroupements émergeraient
probablement.

---

## 6. Question 4 — Prédiction automatique du risque

**Question du directeur :** *« Peut-on prédire automatiquement, dès le
dépistage, si un étudiant sera "à risque" ? Quelle confiance puis-je
accorder à une telle prédiction ? Que se passerait-il si je me trompais ? »*

**Méthode mobilisée : classification supervisée (régression logistique).**
Modèle simple et interprétable, adapté à un contexte médical où la décision
doit rester explicable. Séparation train/test stratifiée (80/20).
Évaluation centrée sur des **métriques orientées risque médical** plutôt que
la seule exactitude : un faux négatif (étudiant réellement à risque non
détecté) est plus grave qu'un faux positif.

### Résultats — seuil de décision standard (0.5)

| | Prédit non-risque | Prédit à-risque |
|---|---|---|
| **Réel non-risque** | 28 | 9 |
| **Réel à-risque** | 8 | 19 |

| Métrique | Valeur |
|---|---|
| Précision (classe à risque) | 0.68 |
| Rappel (classe à risque) | 0.70 |
| F1-score | 0.69 |
| AUC (aire sous la courbe ROC) | 0.82 |
| **Faux négatifs** | **8 sur 64 étudiants test** |

### Effet d'un seuil plus prudent (0.35)

| Métrique | Seuil 0.5 | Seuil 0.35 |
|---|---|---|
| Rappel | 0.70 | **0.89** |
| Précision | 0.68 | 0.53 |
| Faux négatifs | 8 | **3** |

![Classification du risque](figures/q4_classification_risque.png)

### Réponse en langage simple pour le directeur

> Oui, un modèle simple permet de prédire le risque à partir de l'IMC et de
> la tension, avec une **bonne fiabilité** sur cet échantillon (aire sous
> la courbe ROC de 0,82 sur 1,00, nettement au-dessus des 0,50 attendus
> pour un tirage au hasard). Avec les réglages standards, 8 étudiants
> réellement à risque sur les 64 testés passeraient inaperçus — c'est le
> point à surveiller, car le modèle ne connaît que deux mesures parmi tous
> les facteurs qui déterminent le risque réel. En resserrant la vigilance
> du modèle (seuil abaissé à 0,35 au lieu de 0,5), on ramène ces cas
> manqués à seulement 3, au prix de davantage de fausses alertes (la
> précision passe de 0,68 à 0,53). Ce système peut être utilisé avec une
> confiance raisonnable comme outil d'aide au tri initial, mais toute
> décision finale doit rester supervisée par le personnel médical,
> notamment pour les 3 à 8 cas qui resteraient non détectés.

### Discussion des limites

Un AUC de 0,82 est un bon résultat pour un modèle à seulement deux variables,
mais il reste construit sur un échantillon de test relativement petit (64
étudiants) : les métriques rapportées ici (précision, rappel) sont donc
elles-mêmes sensibles au tirage aléatoire du split train/test, et pourraient
varier notablement avec un autre découpage. Le modèle ne capture par
construction qu'une partie du phénomène (une composante du risque dépend,
dans la génération des données, d'un facteur non mesuré) : même avec un bon
score global, il ne faut jamais présenter ce système comme infaillible à
l'échelle individuelle. Une validation croisée à plusieurs répétitions
donnerait une estimation plus stable de la performance réelle avant tout
déploiement en conditions réelles.

---

## 7. Synthèse des méthodes mobilisées

| Question | Bloc de notions | Méthode |
|---|---|---|
| Q1 | Statistique univariée | Moyenne, médiane, écart-type, détection d'outliers par IQR |
| Q2 | Statistique bivariée | Corrélation de Pearson, régression linéaire, analyse des résidus |
| Q3 | Classification non supervisée | K-means, choix de k par score de silhouette |
| Q4 | Classification supervisée | Régression logistique, matrice de confusion, courbe ROC |

Les quatre grands blocs de notions du cours ont ainsi été mobilisés sur
l'ensemble du thème, en cohérence avec les quatre questions posées par le
directeur du service de santé.
