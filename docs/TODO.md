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

- Test 3D cerveau souris sur jonas (mauvais)
- Test 3D cerveau souris sur pipeline (peu concluant)
- Comment choisir les fichier (quel contraste sur 4 dispo ?)
- 2ème test avec scales à partir de 2 (plus homogène -> sensibilité au bruit des méthodes ?)
- Test de l'infuence des scales (sur une seule image)
- Test alpha, beta, gamma (en cours)
- Test scales plusieurs images (en cours)
- Test global sur plusieurs images (code ok mais pas lancé (attente des paramètres optimaux pour le grid search))

---

- [x] Automatiser le lancement du benchmark pour plusieurs images en même temps (ok dans mais mais à tester et lancer)
- [x] Ajouter un fichier avec les différents paramètres utilisées dans chaque expérience (meilleurs paramètres pour le benchmark) (ok mais à vérifier)
- [x] Implémtenter run_all dans le benchmark pour faire une moyenne sur tout les fichiers
- [x] Factoriser la classe viewer
- [x] Factoriser la classe analytics
- [x] Réécrire la logique du main dans run_all de benchmark
- [x] Réécrire la logique de [START] et [END] dans un decorateur spécifique
- [x] Séparer engine et l'étude de l'influence des paramètres dans deux classes distinctes

Chercher comment sélectionner les paramètres du benchmark. Ensuite lancer le benchmark sur toute les images avec toute les images et faire la moyenne des métriques
Lancer le benchmark sur toutes les images 2D possibles
lancer le benchmark pour les images 3D

- [] Résoudre le problème de log benchmark alors que engine
- [] Tester les paramètres alpha, beta, gamma
- [] Faire en sorte de pouvoir modifier les paramètres de main.py avec typer ou autre
- [] Implémenter Sato
- [] Refaire/Revalider les test unitaires pour les fonctions principales. Faire notamment une comparaison avec skimage (et garder les résultats de la comparaison.)
- [] Faire en sorte de généraliser la classe Grid Search pour pouvoir l'utiliser dans d'autres fonction avec d'autres paramètres
- [] Ajouter les cas 2D/3D dans optimizer
- [] Ajouter n_colorsd dans \_get_colors de optimizer.py
- [] Changer la structure du fichier data (raw, labels avec les même nom de fichier et aussi 2d/3d)

Dans le cas 3d faire un crop pour obtenier les meilleurs paramètres plus rapidement
Prendre le min sur les 4 images
