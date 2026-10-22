"""
Point d'entrée principal de l'application FaceGuard - Contrôle d'Accès par Reconnaissance Faciale.
"""

import sys
import os

# Assure que le répertoire racine est dans le PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ui.app import run_app


if __name__ == "__main__":
    print("=" * 70)
    print(" [FACEGUARD] Démarrage du Système de Contrôle d'Accès...")
    print(" - Modèle : VGG-16 (ImageNet, fc1 4096-D)")
    print(" - Détecteur : OpenCV Haar Cascades (Visage Principal)")
    print(" - Interface : Tkinter")
    print("=" * 70)
    run_app()
