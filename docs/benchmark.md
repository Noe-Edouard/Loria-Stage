# Benchmark

## 1. Engine : efficacité du traitement

### 1.1 Objectif

On souhaite trouver les paramètres optimaux pour minimiser le temps de calcul du traitement afin de pouvoir facilement traiter de gros jeu de données et faire de grid search importantes.

### 1.2 Expériences

On effectue chacune des expériences suivantes pour le réhaussement, pour la hessienne et pour le pipeline complet.

- Comparer les temps de calcul avec et sans parallelisation (pour frangi, pour la hessienne et pour le pipeline)
- Comparer les temps de calcul en fonction du nombre d'échelle
- Comparer les temps de calcul en fonction de la taille des chunk
- Comparer les résultats obtenus avec et sans parallelisation pour vérifier que ce sont les mêmes

## 2. Benchmark : précision du réhaussement

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

---

EnhancementBenchmark:
config:
params:
methods
experiments: list[Experiment]
results: list[Results]
method
mean
std

Experiment:
data:
raw
gt
enhanced
segmented
metrics:
mcc
dice
roc
pr
config:

    # Runner
    Pour chaque fichier: (run)
        Charge les données raw/gt (load data)
        # Benhcmark:
        Pour chaque valeurs du paramètre étudié: (run)
            Met à jour le paramètre dans la config de l'experiment (update_param)
            Process data (run_experiment)
            Compute metrics (compute_metrics)
            Ajoute metrics, config, data
            Save experiment (save_experiment)

        Display_metrics (display_metrics)
        Save_metrics (Save metrics)

    compute_mean et std (compute mean)
    ajoute mean std

Ajouter un setup runner ou config runner ?

---

## Objectif final

- Sauvergarder les images

## Benchmark global

### Runner

#### Run

Lance les runs selon le benchmark sélectionné pour toute les images du dataset.

- Récupère tout les chemin vers les images raw et gt du dataset.
- Sélectionne le benchmark
- Appelle la fonction run du benchmark pour chaque image du dataset en parallelisant les calculs.
- Sauvergarde les résultats

#### Analyse

A partir des résultat du benchmark, affiche les visuel (tables, plot, chart, ...) utilent à l'analyse et la comparaison des différentes valeurs du paramètres étudié.

- Charge les résultats
- Remet en forme les résultats pour ne récupérer que les informations essentielles.
- Affiche les plots spécifique au benchmark.

### Benchmmark Base

Classe de base qui permet de construire les différents benchmarks (hessian, enhancement, scales).

#### Run

- Charge l'image raw et la gt associée.
- Met à jour le nom du fichier étudié dans la config
- Pour chaque params :
  - Pour chaque valeur du paramètre:
    - Met à jour la config de l'expérience
    - Lance \_run_experiement pour récupérer les résultats du pipeline voir de la grid search
    - Calcule les métriques par rapport à la segmentation obtenus
    - Crée une expérience
- Crée les figures
- Sauvergarde les figures
- Retourne un dictionnaire {param: {value: Experiment}}

#### Compute_metrics

- Calcule les différentes métriques (dice, mcc, roc, pr) d'après les données réhaussée et segmentée et retourne un objet métrique.

#### Save figures

Donne un nom si la figure n'en a pas et appelle save_figure.

#### Save figure

Affiche la figure si plot_mode et sauvergarde la figure selon le mode sélectionné.

#### Update_config

Met à jour la config selon le paramètre que l'on souhaite étudier.

#### Run_experiment

Lance l'expérience avec les paramètre de la config et lance une grid search si besoin.

#### Create figures

Construit les figures à partir des résultats du run de la forme {param: {value: Experiment}}.

### Benchmark Hessian

#### Init

charge le grid searcher

#### Update_config

Met à jour la méthode du dérivator dans la config.

#### Run_experiment

- Lance un grid search sur les images d'entrée avec les paramètres sélectionnés dans la config.
- Met à jour la config

#### Create figure

- Display Histogram
- Display Configs
- Display Curves (roc, pr)
- Display Views

### Benchmark Enhancement

#### Init

- charge le pipeline

#### Update_config

Met à jour la le paramètre alpha, beta ou gamma. Ou met à jour le paramètre scales en fonction de min/max.

#### Run_experiment

- Lance un pipeline sur les images d'entrée avec les paramètres sélectionnés dans la config.

#### Create figure

- Display mcc_score en fonction des valeurs de Alpha/Beta/Gamma sur 3 subplot d'un même plot.

### Questions

- Est-ce que l'on affiche les résultats intermédiaire au fur et à mesure.
- Est-ce que l'on fait une classe analytics spécifique par benchmark ou on inclu dans la classe benchmark ?
  | On ajoute les plot spécifique dans le run du benchmark et on fait un classe analytics séparée pour les résultats globaux.
- Qu'est-ce que l'on doit sauvergarder exactement comme résultat.
- Doit-on faire une classe de config pour le format des résultats (spécifique à chaque benchamrk ?)
