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

- Comment choisir le gamma quand on parallelize ?

### Autre

- Remplir les fichier `__init__.py`

- Documentation (docs + docstrings, + comment)

---

### Notes

#### Farid

J'ai l'impression que la méthode de Farid permet avant tout d'optimiser l'approximation de la dérivée dans les directions qui ne sont pas celles du repère de l'image (même si il y a une amélioration dans les directions du repère, elle semble significativement moins importante que dans les directions obliques).
A priori, l'améliorations en prenant juste la méthode de Farid et en calculant la hessienne dans les directions de l'image doit être faible ?
Pourrait-on calculer la dérivée dans la direction la plus pertinente (le long de la structure tubulaire ???)

=> balayage angulaire (0, 15, 30, ... 180) => comment sélectionner la meilleures directions ? => 3D pb
=> estimation de la directions principale (via le calcul du gradient) => calcule de la hessienne dans cette direction (couteux !!!)

---

https://www.mdpi.com/2079-9292/12/19/4159

### QUESTION

- Accès article de Sato
- Pas possible de faire la vérification des données 2D avec Jonas
- Code Jonas en C
- Parallelisation ok
- Pipeline ok (check empirique)
- Ajout d'un lisseur gaussiens -> résultats plus cohérents
- Parallelisation du benchmark
- Pas possible de tester le cerveau de souris (trop volumineux)
- Est-ce qu'il ne vaudrait mieux pas ne pas ajouter le filtrage gaussien et implémenter diff finies et gaussian moi même ?
- Quel est le plan pour un article ?

---

But : lancer le benchmark sur toutes les images 2D possibles
lancer le benchmark pour les images 3D

- Ajouter un check sur la plage d'échelle utilisée pour différents nombre d'échelle
- Tester les paramètres alpha, beta, gamma
- Faire les tests sur différentes image (en moyennant ?)
- Résoudre le problème de log benchmark alors que engine
- Automatiser le lancement du benchmark pour plusieurs images en même temps (ok dans mais mais à tester et lancer)
