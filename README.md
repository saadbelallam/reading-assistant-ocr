# Projet de Reconnaissance et Correction de Texte

Ce projet est une pipeline complète de traitement d'image pour la détection, la correction et l'évaluation de texte issu de documents ou photos. Il utilise OpenCV, PyTorch, Tesseract, Transformers (Camembert), etc.

---

## Installation

Aucune installation manuelle de Python requise ! Le script `setup.sh` utilise [`uv`](https://docs.astral.sh/uv/) pour créer un environnement isolé avec la bonne version de Python et toutes les dépendances.

### Étapes :

```bash
make install

```

Ce que ça fait :

Crée un environnement .venv/

Installe toutes les dépendances via pyproject.toml

Télécharge le modèle CamemBERT (~550 Mo) si tu acceptes.

### Commandes Make disponibles

| Commande Make                                     | Description                                                             |
|---------------------------------------------------|-------------------------------------------------------------------------|
| `make install`                                    | Installe toutes les dépendances via `setup.sh`                         |
| `make process-image image=chemin [time=true]`     | Traite une seule image                                                 |
| `make process-all [time=true]`                    | Traite toutes les images dans `tests/images/`                          |
| `make evaluate image=chemin`                      | Évalue la sortie OCR d’une image                                       |
| `make evaluate-all`                               | Évalue tous les résultats OCR dans `tests/results/`                    |
| `make webcam`                                     | Capture une image depuis la webcam et la traite                        |
| `make clean`                                      | Supprime tous les résultats générés                                    |
| `make delete`                                     | Supprime l’environnement `.venv/`, le modèle CamemBERT et les résultats |
| `make help`                                       | Affiche un résumé de toutes les commandes disponibles                  |






