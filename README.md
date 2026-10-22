# FaceGuard System - Contrôle d'Accès par Reconnaissance Faciale (VGG-16)

Système complet de contrôle d'accès biométrique par reconnaissance faciale développé en Python selon une approche itérative en 4 jalons.

## 🛠️ Stack Technique
- **Python** (3.9+)
- **OpenCV** : Capture de flux vidéo temps réel et prétraitements.
- **Haar Cascades** : Détection et recadrage du visage principal.
- **TensorFlow / Keras (VGG-16)** : Extraction de vecteurs d'embeddings (4096 dimensions via la couche dense `fc1` pré-entraînée sur ImageNet).
- **Similarité Cosinus** : Mesure de distance/similarité biométrique avec seuil ajustable.
- **Stockage JSON** : Base de données locale des signatures d'utilisateurs (`data/users.json`).
- **Tkinter** : Interface graphique interactive avec retour visuel d'accès (Vert/Rouge).

---

## 🚀 Jalons du Projet

- [x] **Jalon 1 : Moteur d'extraction de caractéristiques (VGG-16)**
  - Prétraitement $224 \times 224 \times 3$ et normalisation ImageNet.
  - Extraction de descripteurs profonds 4096-D (couche `fc1`).
  - Normalisation L2 et similarité cosinus.
  - Validation par banc de tests unitaires (`tests/test_milestone1.py`).
- [x] **Jalon 2 : Détection en temps réel & isolation du visage**
  - Intégration de Haar Cascade frontal (`models/haarcascade_frontalface_default.xml`).
  - Ciblage automatique du visage principal (surface maximale).
  - Découpage sécurisé avec marge de sécurité (*padding*).
  - Script temps réel webcam avec PiP et affichage FPS (`tests/test_milestone2_webcam.py`).
  - Validation par banc de tests unitaires et intégration (`tests/test_milestone2.py`).
- [x] **Jalon 3 : Module d'enrôlement & comparaison avec seuil**
  - Base de données locale JSON persistante (`core/database.py`).
  - Algorithme de comparaison cosinus avec seuil de similarité (`core/matcher.py`).
  - Système double d'enrôlement : capture webcam directe ou fichier image (`core/enrollment.py`).
  - Statuts d'accès formels : **« Autorisée »** vs **« Refusée »**.
  - Validation par banc de tests unitaires et fonctionnels (`tests/test_milestone3.py`).
- [ ] **Jalon 4 : Interface utilisateur finale (Tkinter)**

---

## 🧪 Exécution des tests

### Tests unitaires Jalon 1 (Extraction VGG-16) :
```bash
python tests/test_milestone1.py
```

### Tests unitaires Jalon 2 (Détection & isolation) :
```bash
python tests/test_milestone2.py
```

### Démo flux webcam temps réel (Jalon 2) :
```bash
python tests/test_milestone2_webcam.py
```

### Tests unitaires Jalon 3 (Enrôlement, JSON & matching) :
```bash
python tests/test_milestone3.py
```
