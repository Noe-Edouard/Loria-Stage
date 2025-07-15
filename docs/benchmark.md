# Benchmark

## 1. Benchmark efficacité du traitement

### 1.1 Objectif

On souhaite trouver les paramètres optimaux pour minimiser le temps de calcul du traitement afin de pouvoir facilement traiter de gros jeu de données et faire de grid search importantes.

### 1.2 Expériences

On effectue chacune des expériences suivantes pour le réhaussement, pour la hessienne et pour le pipeline complet.

- Comparer les temps de calcul avec et sans parallelisation (pour frangi, pour la hessienne et pour le pipeline)
- Comparer les temps de calcul en fonction du nombre d'échelle
- Comparer les temps de calcul en fonction de la taille des chunk
- Comparer les résultats obtenus avec et sans parallelisation pour vérifier que ce sont les mêmes

## 2. Benchmark précision du réhaussement

### 2.1 Objectif

On souhaite évaluer l'influence de la méthode de différentiation dicrète sur le résultat du réhaussement vasculaire obtenu par la méthode de Frangi.

### 2.2 Méthode

Pour cela, on peut envisager différentes méthodes :

- comparer le résultat du réhaussement avec les mêmes paramètres.
- Comparer le meilleurs résultats obtenu avec les paramètres optimisés (obtenus par grid search)

### 2.3 Affichage

Pour chacune des expériences, il faut afficher :

- Histogram
- Coubre ROC / PR (comment visualiser le seuil ?)
- Comparaisons visuelles de l’image réhaussée par méthode (avec faux positifs, faux négatifs, ... en rouge, vert, bleu)
- Valeurs des scores dice, mcc, roc, pr dans un tableau.

### 2.4 Métriques

#### 2.4.1 ROC (Receiver Operating Characteristic) Curve

##### Description

Évalue la capacité d’un modèle à discriminer entre les classes (vaisseau vs fond), à tous les seuils possibles.
Si classes déséquilibrés on utilise la courbe Precision-Recall

##### Calcul

- On calcule pour tout les seuil de binarisation les True Positive Rate (TPR=(FP)/(FP+TN)) = les vaisseaux et le nombre de False Positive Rate (FPR=(TP)/(TP+FN)) = le fond
- On trace la courbe TPR(FPR)
- On calcule l’aire sous la courbe (AUC = Area Under Curve)

##### Interprétation

- Le seuil idéal correspond au point dans le coin supérieur gauche.
- Plus la courbe est diagonale, plus le modèle est proche de l'aléatoire.
- Sous la diagonale, le modèle est inversé.V
- AUC-ROC proche de 1 = bonne discrimination, 0.5 = aléatoire

#### 2.4.2 PR (Precision-Recall) Curve

##### Description

Mieux adaptée que la ROC quand la classe positive est rare.

##### Calcul

- On calcule pour tout les seuils la Précision = TP / (TP + FP) et le Rappel = TP / (TP + FN)
- On trace la courbe Précision(Rappel)
- On calcule l'aire sous la courbe (AUC-PR)

##### Interprétation

- proche de 1 = bonne capacité à récupérer les vaisseaux avec peu de faux positifs.

#### 2.4.3 DICE Coefficient

##### Description

Evalue la similarité entre les masques segmentés et les masques de référence.

Sensible au désalignement spatial

##### Calcul

DICE = (2*TP) / (2*TP + FP + FN)

##### Interprétation

- DICE = 1 => correspondance parfaite
- DICE = 0 => aucune correspondance

#### 2.4.4 MCC (Matthews Correlation Coefficient)

##### Description

Evalue la qualité de la classification binaire.

Très robuste même en cas de déséquilibre de classes.

##### Calcul

- MCC = TP _ TN - FP _ FN / (sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)})

##### Interprétation

- MCC = 1 : prédiction parfaite
- MCC = 0 : pas mieux que le hasard
- MCC = -1 : totalement incorrect
