# FaceGuard System - Contrôle d'Accès par Reconnaissance Faciale (VGG-16)

Système complet de contrôle d'accès biométrique par reconnaissance faciale développé en Python selon une approche itérative en 4 jalons stricts.

---

## 🛠️ Stack Technique Imposée
- **Langage** : Python 3.9+
- **Capture vidéo & Prétraitement** : OpenCV (`opencv-python 4.x`)
- **Détection & Recadrage de visage** : Haar Cascades (avec isolation du visage principal et marge de sécurité)
- **Deep Learning / Extraction de Caractéristiques** : TensorFlow / Keras avec **VGG-16** pré-entraîné sur ImageNet (Option B : couche dense intermédiaire `fc1`, vecteur d'embedding de 4096 dimensions normalisé L2)
- **Métrique Biométrique** : Similarité Cosinus avec seuil ajustable en temps réel
- **Stockage Local** : Fichier JSON (`data/users.json`) pour l'enregistrement des signatures et profils
- **Interface Utilisateur** : Tkinter avec retour visuel dynamique (Vert/Rouge/Gris), slider de sensibilité et gestionnaire d'enrôlement

---

## 🚀 Jalons du Projet (100% Terminés)

- [x] **Jalon 1 : Moteur d'extraction de caractéristiques (VGG-16)**
  - Prétraitement strict $224 \times 224 \times 3$, conversion RGB et normalisation ImageNet.
  - Extraction de vecteurs de descripteurs profonds 4096-D (`fc1`).
  - Normalisation L2 et calcul de similarité cosinus.
  - Validation par banc de tests unitaires ([`tests/test_milestone1.py`](tests/test_milestone1.py)).

- [x] **Jalon 2 : Détection en temps réel & isolation du visage**
  - Classificateur Haar Cascade frontal ([`models/haarcascade_frontalface_default.xml`](models/haarcascade_frontalface_default.xml)).
  - Sélection automatique du visage principal (aire maximale $w \times h$).
  - Découpage sécurisé du visage avec marge (*padding* $+15\%$).
  - Script webcam de test avec incrustation PiP et FPS ([`tests/test_milestone2_webcam.py`](tests/test_milestone2_webcam.py)).
  - Validation par banc de tests unitaires ([`tests/test_milestone2.py`](tests/test_milestone2.py)).

- [x] **Jalon 3 : Module d'enrôlement & comparaison avec seuil**
  - Base de données locale persistante JSON ([`core/database.py`](core/database.py)).
  - Algorithme de comparaison biométrique cosinus avec seuil ([`core/matcher.py`](core/matcher.py)).
  - Système double d'enrôlement : capture webcam directe ou importation de fichier ([`core/enrollment.py`](core/enrollment.py)).
  - Statuts formels : **« Autorisée »** vs **« Refusée »**.
  - Validation par banc de tests unitaires ([`tests/test_milestone3.py`](tests/test_milestone3.py)).

- [x] **Jalon 4 : Interface utilisateur finale (Tkinter)**
  - Assemblage modulaire complet dans [`ui/app.py`](ui/app.py).
  - Flux vidéo webcam direct 30 FPS avec tracking temps réel et PiP.
  - Bannière visuelle d'accès : **VERT** (Autorisée), **ROUGE** (Refusée), **GRIS** (En attente).
  - Curseur en direct (*Slider*) pour ajuster le seuil de sensibilité.
  - Double option d'enrôlement : bouton capture webcam et bouton import image.
  - Tableau des utilisateurs enregistrés avec suppression et actualisation.
  - Validation par banc de test UI ([`tests/test_milestone4_ui.py`](tests/test_milestone4_ui.py)).

---

## 🖥️ Lancement de l'Application

Pour démarrer l'application complète avec l'interface graphique Tkinter :
```bash
python main.py
```

---

## 🧪 Exécution de l'Ensemble des Tests

Chaque jalon possède son propre script de test unitaire et d'intégration :

```bash
# Test Jalon 1 : Moteur VGG-16 (fc1 4096-D, preprocessing 224x224)
python tests/test_milestone1.py

# Test Jalon 2 : Détection Haar Cascade & isolation visage principal
python tests/test_milestone2.py

# Démo Webcam temps réel (Jalon 2)
python tests/test_milestone2_webcam.py

# Test Jalon 3 : Base JSON, Matcher biométrique et Enrôlement double
python tests/test_milestone3.py

# Test Jalon 4 : Interface graphique Tkinter
python tests/test_milestone4_ui.py
```

---

## 📁 Structure du Projet

```text
face-guard-system/
│
├── core/                        # Cœur fonctionnel et algorithmique
│   ├── __init__.py
│   ├── extractor.py             # Jalon 1 : Extracteur VGG-16 (fc1 4096-D)
│   ├── detector.py              # Jalon 2 : Détecteur Haar Cascade (visage principal)
│   ├── database.py              # Jalon 3 : Base de données locale JSON
│   ├── matcher.py               # Jalon 3 : Comparaison cosinus & décision seuil
│   └── enrollment.py            # Jalon 3 : Gestionnaire d'enrôlement double
│
├── ui/                          # Interface utilisateur graphique
│   ├── __init__.py
│   └── app.py                   # Jalon 4 : Application Tkinter
│
├── models/                      # Modèles pré-entraînés
│   └── haarcascade_frontalface_default.xml
│
├── data/                        # Stockage local persistant
│   ├── users.json               # Fichier JSON des signatures utilisateurs
│   └── enrolled/                # Photos des visages enrôlés
│
├── tests/                       # Bancs de tests automatisés par jalon
│   ├── test_milestone1.py
│   ├── test_milestone2.py
│   ├── test_milestone2_webcam.py
│   ├── test_milestone3.py
│   └── test_milestone4_ui.py
│
├── main.py                      # Point d'entrée de l'application
├── requirements.txt             # Dépendances Python
└── README.md                    # Documentation complète
```
