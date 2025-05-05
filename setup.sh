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

#Vérifie si le modèle est déjà présent
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo ""
  echo "Le modèle Camembert n'est pas encore téléchargé."
  echo "Ce téléchargement va prendre environ 550 Mo."
  read -p "Souhaitez-vous télécharger le modèle maintenant ? (y/n) " confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Téléchargement du modèle Camembert..."
    python download.py
  else
    echo "Téléchargement du modèle annulé. Certaines fonctionnalités NLP ne seront pas disponibles tant qu’il n’est pas téléchargé."
  fi
else
  echo "Modèle Camembert déjà présent."
fi

echo ""
echo "Installation terminée avec succès."
echo "Pour activer l'environnement virtuel : source $VENV_DIR/bin/activate"
echo "Pour traiter une image : make process-image image=test.jpeg RESULT_DIR=results/"
