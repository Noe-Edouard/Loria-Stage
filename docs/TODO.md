### Tests

- Test unitaire estimator
- Modifier la fonction de test de farid (comparer aux résultats de l'article)
- Uniformiser les tests sur les images réelles (comparaison avec frangi de skimage)

### Benchmark

- Implémenter Sato
- Images bruitées
- Différents type d'image

- Tester précision en foncion du nombre d'échelle (faire un test 2D et 3D)

### Config

- Ajouter la possibilité de changer la config avec typer
- Comment choisir le gamma quand on parallelize ?

### Autre

- Remplir les fichier `__init__.py`

- Documentation (docs + docstrings, + comment)

---

### QUESTIONS

#### Farid

J'ai l'impression que la méthode de Farid permet avant tout d'optimiser l'approximation de la dérivée dans les directions qui ne sont pas celles du repère de l'image (même si il y a une amélioration dans les directions du repère, elle semble significativement moins importante que dans les directions obliques).
A priori, l'améliorations en prenant juste la méthode de Farid et en calculant la hessienne dans les directions de l'image doit être faible ?
Pourrait-on calculer la dérivée dans la direction la plus pertinente (le long de la structure tubulaire ???)

=> balayage angulaire (0, 15, 30, ... 180) => comment sélectionner la meilleures directions ? => 3D pb
=> estimation de la directions principale (via le calcul du gradient) => calcule de la hessienne dans cette direction (couteux !!!)

---

https://www.mdpi.com/2079-9292/12/19/4159

- Ecrire rapport de stage
-
