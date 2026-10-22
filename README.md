# FaceGuard System - Contrôle d'Accès par Reconnaissance Faciale (VGG-16)

Système complet de contrôle d'accès biométrique par reconnaissance faciale développé en Python selon une approche itérative en 4 jalons.

## 🛠️ Stack Technique
- **Python** (3.9+)
- **OpenCV** : Capture de flux vidéo temps réel et prétraitements.
- **Haar Cascades** : Détection et recadrage du visage principal.
- **TensorFlow / Keras (VGG-16)** : Extraction de vecteurs d'embeddings (4096 dimensions via la couche dense `fc1` pré-entraînée sur ImageNet).
- **Similarité Cosinus** : Mesure de distance/similarité biométrique avec seuil ajustable.
- **Stockage JSON** : Base de données locale des signatures d'utilisateurs.
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
- [ ] **Jalon 3 : Module d'enrôlement & comparaison avec seuil**
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
*(Appuyez sur `q` pour quitter ou `s` pour enregistrer un visage capturé)*
