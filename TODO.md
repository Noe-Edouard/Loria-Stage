## Roadmap - Pipeline de Réhaussement Vasculaire

### 1. Structure du projet

- [ ] Organiser la structure du repo (modules, scripts, notebooks, data, tests, docs)
- [ ] Créer un environnement Python reproductible `requirements.txt`
- [ ] Ajouter un fichier `README.md` explicatif
- [ ] Mettre en place une arborescence pour les logs et les résultats
- [ ] Centraliser la configuration avec un fichier `config.yaml` ou `.json`

### 2. Prétraitement

#### 2.1 Chargement (fichier loader.py)

- [ ] Détecter automatiquement le format d’image (.tif, .nii, .raw, .png, .jpg...)
- [ ] Retourner l'image au format (X, Y, Z) ou (X, Y) selon le cas (2D ou 3D)
- [ ] Créer des loaders multi-formats (nibabel, SimpleITK, skimage, PIL, etc.)
- [ ] Lors de la sauvegarde d'une image le format doit être `.nii.gz` quel qu'était le format initial.
- [ ] Gérer les traitements 2D et 3D dans un cadre unifié mais distinct
- [ ] Demander une validation pour traiter une image déjà traitée

#### 2.2 Chunking (fichier chunker.py)

- [ ] Implémenter un système de chunking pour les images 3D lourdes
- [ ] Ajouter la possibilité de crop l'image au centre en choississant la dimension voulue. Gérer le cas ou la dimension demandé est None (taille max) et ou la dimension demandé est supperieur à la dimension de l'image dans la direction précisée.

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

- Mettre en place les logs
- Externaliser les paramètres dans un fichier (regarder le code de max)

##### 5.1 En Fonction du temps d'exécution

- [ ] Etudier l'influence de différents paramètres sur le temps d'éxécution. L'objectif est de trouver les paramètres qui optimisent les temps d'éxécutions.
  - [ ] influence de la taille des chunks (avec pour référence le temps sur CPU)
  - [ ] influence de la taille de l'image (avec et sans parallélisation)
  - [ ] influence de la parallélisation (temps d'éxécution du traitement pour différentes taille de chunk % du coté)
- [ ] Comaprer les temps de calcul et les réponses visuelles sans réhaussement

#### 1.2 En fonction de la précision

Etude de l'influence de différents paramètres et méthodes de calcul/filtrage sur le temps d'éxécution. L'objectif est de trouver les paramètres et les méthode qui optimisent la reconnaissance des vaisseaux.

- influence de alpha, beta, gamma
- influence de la méthode de segmentation
- influence de la méthode de différentiation

#### 5.1 Méthodes de différentiation

- [ ] Comparer l'influence de chaque méthode sur le réhaussement

#### 5.2 Méthode de réhaussement

- [ ] Identifier et implémenter des métriques pertinentes (SNR, CNR, Dice, AUC…)
- [ ] Utiliser les masques pour calculer les métriques supervisées
- [ ] Créer des visualisations comparatives (boxplot, heatmap…)
- [ ] Ajouter une CLI pour lancer le benchmark facilement

### 7. Visualisation

- [ ] Visualisation 2D avec `matplotlib` ou `napari`
- [ ] Visualisation 3D interactive (napari, pyvista, vedo…)
- [ ] Vue synchronisée avant/après avec overlays
- [ ] Visualiseur `.nii` en notebook et terminal

### 8. Analyse exploratoire des données

- [ ] Calculer des stats globales (intensités, dimensions…)
- [ ] Afficher des histogrammes et densités
- [ ] Extraire et afficher les métadonnées disponibles

### 9. Tests & Documentation

- [ ] Écrire des tests unitaires avec `pytest`
- [ ] Ajouter des tests d’intégration sur sous-ensemble d’images
- [ ] Générer une documentation technique (mkdocs / Sphinx)
- [ ] Fournir des notebooks explicatifs pour chaque module

### 10. Performance & Production

- [ ] Ajouter une parallélisation des traitements (multiprocessing / dask)
- [ ] Gérer les gros volumes avec lecture mémoire optimisée
- [ ] Ajouter un cache de résultats
- [ ] Créer un mode debug rapide avec sous-échantillonnage

### 11. Interface utilisateur

- [ ] Créer une CLI pour exécuter chaque étape du pipeline
- [ ] Gérer les options en ligne de commande (argparse / typer)
- [ ] Ajouter des logs détaillés (logging / rich)

### 12. Collaboration & Déploiement

- [ ] Structurer le dépôt Git avec conventions (`main`, `dev`, `feature/*`)
- [ ] Écrire un guide d'installation (README / INSTALL.md)
- [ ] Gérer les dépendances (`requirements.txt`, `environment.yml`)
- [ ] Ajouter un `setup.py` ou `pyproject.toml` pour le packaging

### Taches non prioritaire

- [ ] Normaliser l'image (z-score) avant le traitement ? Etudier l'influence de la normalisation ?
- [ ] Ajouter une option de débruitage (Gaussian, Median, etc.)

- [ ] Ajouter une option de binarisation pour la reconstruction
