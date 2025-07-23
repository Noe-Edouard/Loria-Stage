- Les résultats obtenus avec les filtres brutes étaient mauvais donc on décide d'ajouter un filtrage gaussien qui permet de prendre en compte les différentes échelles

- Attention à bien sélectionner black_ridges lors du lancement du benchmark

- Lorsqu'on utilise le pipeline la parallelisation pour les images 3D on ne peut pas utiliser gamma = Null car sinon il y a un gamma différent par chunk. Il faut donc fixer le gamma et ne pas utiliser null dans les paramètres du benchmark.

- Parralelization du benchmark, du traitement des chunk pour les images 3D
