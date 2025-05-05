#!/bin/bash
set -e

PYTHON_VERSION=$(cat .python-version 2>/dev/null || echo "3.11.8")
VENV_DIR=".venv"
MODEL_DIR="models/camembert-base"

echo "Installation du projet (Python $PYTHON_VERSION via uv)"

# Vérifie que uv est installé
echo "Vérification de uv..."
if ! command -v uv &> /dev/null; then
  echo "uv non trouvé. Installation via pip..."
  python3 -m pip install --upgrade pip
  python3 -m pip install uv
else
  echo "uv est déjà installé"
fi

# Crée l'environnement virtuel avec la version Python spécifiée (via .python-version si présent)
echo "Création de l'environnement virtuel avec Python $PYTHON_VERSION..."
uv venv "$VENV_DIR"

# Active le venv
source "$VENV_DIR/bin/activate"

# Installe les dépendances à partir de pyproject.toml
echo "Installation des dépendances..."
uv sync

echo ""
echo "Installation terminée avec succès."
echo "Pour activer l'environnement virtuel : source $VENV_DIR/bin/activate"
echo "Pour traiter une image : make process-image image=test.jpeg RESULT_DIR=results/"
