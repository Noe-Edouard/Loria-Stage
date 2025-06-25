## TO DO - Pipeline de Réhaussement Vasculaire

### 1. Structure du projet

- [x] Organiser la structure du projet
- [ ] Mettre en place une arborescence pour les logs et les résultats
- [ ] Centraliser la configuration avec un fichier `config.yaml` ou `.json`
- [ ] Écrire des tests unitaires avec `pytest`

### 2. Prétraitement

#### 2.1 Chargement (fichier loader.py)

- [x] Détecter automatiquement le format d’image (.tif, .nii, .raw, .png, .jpg...)
- [x] Retourner l'image au format (X, Y, Z) ou (X, Y) selon le cas (2D ou 3D)
- [x] Créer des loaders multi-formats (nibabel, SimpleITK, skimage, PIL, etc.)
- [x] Lors de la sauvegarde d'une image le format doit être `.nii.gz` quel qu'était le format initial.
- [ ] Gérer les traitements 2D et 3D dans un cadre unifié mais distinct
- [ ] Demander une validation pour traiter une image déjà traitée

#### 2.2 Chunking (fichier chunker.py)

- [x] Implémenter un système de chunking pour les images 3D lourdes
- [x] Ajouter la possibilité de crop l'image au centre en choississant la dimension voulue. Gérer le cas ou la dimension demandé est None (taille max) et ou la dimension demandé est supperieur à la dimension de l'image dans la direction précisée (sous-échantillonnage).
- [x] Mettre en place la parrallélisation des calculs
- [ ] Vérifier que l'on obtient bien les même résultat avec et sans parallélisation pour les données réelles

### 3. Méthodes de différentiation (Hessienne)

#### 3.1 Méthode de base par convolution (référence)

- [ ] Implémenter une Hessienne de base à la main (convolution)
- [ ] Regarder comment est codée la méthode de `skimage`

#### 3.2 Méthode de Farid

- [ ] Analyser et comprendre la méthode de Farid
- [ ] Implémenter la méthode de Farid
- [ ] Ajouter un tests unitaire

#### 3.3 Méthode de Hast

- [ ] Analyser et comprendre la méthode de Hast
- [ ] Implémenter la méthode de Hast
- [ ] Ajouter un tests unitaire

### 4. Réhaussement vasculaire

#### 4.1 Méthode de Frangi

- [ ] Implémenter la méthode de Frangi from scratch
- [ ] Mettre en place un test unitaire pour valider l'implémentation
- [ ] Comparer la méthode avec la méthode `frangi` du module `skimage.filters`
- [ ] Comparer la méthode avec les résultats des articles
- [ ] Normaliser la sortie pour la comparaison

#### 4.2 Autres méthodes

- [ ] Ajouter des hooks pour d'autres filtres (Meijering, Sato, Jerman…)

### 5. Benchmark

#### 5.1 Strcuture

- [ ] Créer des visualisations comparatives (boxplot, heatmap…)
- [ ] Ajouter une CLI pour lancer le benchmark facilement
- [ ] Mettre en place les logs
- [ ] Externaliser les paramètres dans un fichier (regarder le code de max)
- [ ] Identifier et implémenter des métriques pertinentes (SNR, CNR, Dice, AUC…)

##### 5.2 En Fonction du temps d'exécution

- [ ] Etudier l'influence de différents paramètres sur le temps d'éxécution. L'objectif est de trouver les paramètres qui optimisent les temps d'éxécutions.
  - [ ] influence de la taille des chunks (avec pour référence le temps sur CPU)
  - [ ] influence de la taille de l'image (avec et sans parallélisation)
  - [ ] influence de la parallélisation (temps d'éxécution du traitement pour différentes taille de chunk % du coté)
- [ ] Comaprer les temps de calcul et les réponses visuelles sans réhaussement

#### 5.3 En fonction de la précision

- [ ] Etudier l'influence de différents paramètres et méthodes de calcul/filtrage sur la précision post-filtrage. L'objectif est de trouver les paramètres et les méthodes qui optimisent la reconnaissance des vaisseaux.
  - [ ] influence de alpha, beta, gamma
  - [ ] influence de la méthode de segmentation
  - [ ] influence de la méthode de différentiation

### 6. Analytics

#### 6.1 Visualisation

- [x] Visualisation 2D et 3D avec `matplotlib`
- [ ] Améliorer la visualisation avec `napari`, `pyvista`, `vedo`
- [ ] Vue synchronisée avant/après
- [ ] Regarder s'il y a une API pour 3D Slicer pour pouvoir facilement faire des visualisations

### 6.2 Analyse des données

- [ ] Calculer des stats globales (intensités, dimensions…)
- [ ] Afficher des histogrammes et densités
- [ ] Extraire et afficher les métadonnées disponibles

### 7. Utilisateur

#### 7.1 Documentation

- [ ] Créer un environnement Python reproductible `requirements.txt`
- [ ] Ajouter un fichier `README.md` explicatif

#### 7.2 Interface utilisateur

- [ ] Créer une CLI pour exécuter chaque étape du pipeline
- [ ] Gérer les options en ligne de commande (argparse / typer)
- [ ] Ajouter des logs détaillés (logging / rich)

### Taches non prioritaire

- [ ] Normaliser l'image (z-score) avant le traitement ? Etudier l'influence de la normalisation ?
- [ ] Ajouter une option de débruitage (Gaussian, Median, etc.)
- [ ] Ajouter une option de binarisation pour la reconstruction

### Aujourd'hui

#### Pipeline

- [x] Normalisation
- [x] Renommer les données (appps1_cc)
- [x] Regarder Slicer 3D
- [x] Renommer le fichier en chunker.py
- [ ] Regarder métriques
- [ ] Test chunkage
- [ ] Check parallélisation/sequentiel même résultat sur vrai données
- [ ] Modifier l'appel direct à frangi filter dans computational_time en appel à apply filter de pipeline
- [ ] Trouver un autre nom pour les fichier de benchmark (comptational time est affreux)

#### Logger

- [ ] Ajouter la possibilité de print des données externes dans le logger pour ne pas avoir à le faire dans les fonctoin.
- [ ] AJouter les barres tqdm dans le logger
- [ ] Transfert de argparse vers typer
- [ ] Vérifier qu'il n'y a pas trop de conflit entre les logs (pas de répétition)
- [ ] Modifier le print des arg dans debug_logger (dans le cas du volume on ne veut pas afficher tout le tableau)
- [ ] Enlever les log et les print dans la plupart des fonction et les changer par le logger

#### Autre

- [ ] Réécrire les taches dans notion en ajoutant une page de visu par section (benchmark, pipeline, différentiation...)
