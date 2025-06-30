# 📁 Structure du Projet (Template Modulaire)

Ce document présente une organisation modulaire, claire et évolutive pour un projet Python orienté pipeline, machine learning ou traitement de données.

---

## Noms de dossiers

- `src/` : Contient le code source principal (logique métier, pipeline, modules).

- `utils/` : Fonctions utilitaires réutilisables (logger, helpers, etc.).

- `data/` : Données brutes ou d’entrée, souvent en lecture seule.

- `outputs/` : Résultats générés : figures, fichiers, prédictions, rapports.

- `logs/` : Logs d’exécution horodatés, structurés.

- `models/` : Modèles entraînés sauvegardés (`.pt`, `.pkl`, etc.).

- `tests/` : Tests unitaires et d’intégration, fixtures.

- `configs/` : Paramètres de configuration (YAML, JSON, TOML, ou scripts Python).

- `docs/` : Documentation technique, diagrammes, guides, fichiers Markdown.

---

## Noms de fichiers usuels

### Utilitaires (`utils/` ou `src/utils/`)

- `logger.py` : Configuration et gestion des logs
- `chunker.py` : Découpage de données (batch, séquences, chunks)
- `fetcher.py` : Récupération de données depuis API, web, cloud
- `helper.py` : Fonctions utilitaires diverses, helpers
- `loader.py` : Chargement des données
- `saver.py` : Sauvegarde des données, modèles, résultats
- `writer.py` : Écriture personnalisée (CSV, JSON, autres)
- `reader.py` : Lecture personnalisée de formats variés
- `parser.py` : Analyse et transformation de formats complexes (HTML, JSON, XML)
- `splitter.py` : Séparation des jeux de données (train/test/val)
- `scheduler.py` : Planification et gestion de tâches ou événements
- `visualizer.py` : Fonctions de visualisation et affichage graphique
- `plotter.py` : Création de graphiques statiques ou interactifs
- `builder.py` : Construction d’objets complexes (modèles, pipelines)
- `normalizer.py` : Normalisation ou standardisation des données
- `formatter.py` : Mise en forme des données, des logs ou des textes
- `timer.py` : Mesure du temps d’exécution, profilage
- `metrics.py` : Calcul et définition des métriques d’évaluation
- `pipeline.py` : Pipeline principal ou orchestrateur
- `train.py` / `trainer.py` : Entraînement des modèles
- `test.py` / `tester.py` : Évaluation ou test des modèles
- `runner.py` : Lanceur du pipeline complet

> **Convention recommandée** : utiliser des noms en forme de `verbe_attribut.py`,  
> par exemple `compute_hessian.py`, `evaluate_model.py`.

---

## Fichiers généraux

- `README.md` : Documentation principale du projet
- `LICENSE` ou `LICENSE.md` : Licence du projet
- `CHANGELOG.md` : Historique des versions et modifications
- `CONTRIBUTING.md` : Guide pour contributeurs
- `TODO.md` : Liste des tâches à faire
- `.gitignore` : Fichiers et dossiers à ignorer par Git
- `requirements.txt` : Liste des dépendances Python
- `setup.py` : Script d’installation (package Python)
- `config.py` ou fichiers `*.yaml`, `*.json` : Paramètres de configuration
- `main.py` : Point d’entrée principal du programme
- `__init__.py` : Marque un dossier comme package Python

e ?
