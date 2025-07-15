### Utils

- IO > Augmenter le nombre d'extensions supportées par la fonction save, améliorer la fonction pour inclure les métadonnées

### Tests

- Test unitaire estimator

- Modifier la fonction de test de farid (comparer aux résultats de l'article)

- Uniformiser les tests sur les images réelles (comparaison avec frangi de skimage)

### Derivator

- Ajouter sigma dans farid derivator

- Normaliser les sortie du réhaussement avec une sigmoide plutot que min max pour conserver les probabilités ??? (ROC et PR ont besoin de probas (je crois))

- Uniformiser les méthodes dans derivator pour éviter d'avoir à les réécrire (les écrire dans la config directement)

### Enhacemnent

- Résoudre les problèmes de filtre gaussien dans la méthode de farid

- Supprimer la normalisation dans les tests de frangi.

### Benchmark

#### Enhancement

- Renommer ConfigParams et Config en Pipeline config > Gérer la config du pipline et du benchmark séparément

- Modifier la fonction de segmentation pour optimiser le MCC

- Implémenter le grid search dans le benchmark

#### Computational time

- Lancer le benchmark de computational time sur le super ordi

### Config

- Séparer la logique de la config entre le pipeline et le benchmark (séparer la config du setup du pipeline et la config du run)
- Faire en sorte de pouvoir lancer différents run avec une même config de pipeline

- Ajouter la possibilité de changer la config avec typer

- Changer la logique de sauvegarde (pour l'instant tout est dans le même fichier, c'est le brodel, ajouter une clé (avec les paramètres))

- Ajouter la logique de sauvegarde à la fin du run

- séparer la méthode de skimage en deux méthodes (use et pas use) surpprimer le paramètre use_gaussian dérivatibe dans les configs

- Comment choisir le gamma quand on parallelize ?

### Autre

- Remplir les fichier `__init__.py`

- Documentation (docs + docstrings, + comment)

---

### QUESTIONS

- Il y a un problème d'échelle avec la manière avec laquelle on calcule la dérivée
- Comment prendre en compte l'échelle sans le filtre gaussien

D'où vient l'expression de la contrainte de steerabilité ???

Métriques ok ?

L'implémentation fonctionne sur un exemple simple mais problème sur le test.
Le problème semble venir de l'utilisation du filtre gaussien

J'ai l'impression que la méthode de Farid permet avant tout d'optimiser l'approximation de la dérivée dans les directions qui ne sont pas celles du repère de l'image (même si il y a une amélioration dans les directions du repère, elle semble significativement moins importante que dans les directions obliques).
A priori, l'améliorations en prenant juste la méthode de Farid et en calculant la hessienne dans les directions de l'image doit être faible ?
Pourrait-on calculer la dérivée dans la direction la plus pertinente (le long de la structure tubulaire ???)

=> balayage angulaire (0, 15, 30, ... 180) => comment sélectionner la meilleures directions ? => 3D pb
=> estimation de la directions principale (via le calcul du gradient) => calcule de la hessienne dans cette direction (couteux !!!)

---

https://www.mdpi.com/2079-9292/12/19/4159

### Notes

- [x] Ne pas normaliser
- [x] Regarder le résultat de la dérivée avant de passer à la suite
- [x] Regarder l'implémentation de Hast (notamment pour farid)
- [x] Vérifier l'implémentation de la double convolution 1d en 3d
- [x] Ralentir les animations
- [~] Récupérer la ground truth du fichier test.nii
- [~] Mettre en place un benchmark sur L'ENSEMBLE DES PARAMETRES
- [ ] Voir comment prendre en compte l'échelle dans la hessiennes perosnalisées (hast, farid)

- plot image de référence + display min max
- plot image référence + bruit (gaussien)
- Dérivées 1 + display min max
- Dérivées 2 + display min max

- Erreur absolue par rapport à la référence + display min max
- Erreur quadratique par rapport à la référence + display min max
- Regarder d'autre images qu'une gaussienne
- Normaliser en entrée

- Regarder l'energie ddes filtres par rapport au facteur
- Evolution du min, max en fonction de l'échelle
- N dépende de l'ordre de la convolution

- Reagreder la doc de skimage pour le calcul de la dérviée

- Regarder différentes types de fonction (sinc, sinus grating, ...)
- MSE

---

afficher meilleur score pour chaque grid search
Afficher le temps de chaque grid search
Afficher le temps du benchmark
Afficher le temps du pipeline
Ajouter skimage default et gaussian à la place de default.
supprimer les critical
Afficher le threshold à la fin du grid search
Ajouter une ligne best methods dans le tableau
Enlever les normalisation (test)
supprimer et résoudre les test.log.log

- Regarder le bruit
- Ajouter un filtre gaussien
