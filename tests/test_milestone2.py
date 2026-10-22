"""
Script de test unitaire et d'intégration pour le Jalon 2 : Détection et isolation du visage.
"""

import sys
import os
import unittest
import numpy as np
import cv2

# Ajout de la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detector import FaceDetector
from core.extractor import VGG16FeatureExtractor


class TestMilestone2FaceDetector(unittest.TestCase):
    """
    Validation unitaire des spécifications du Jalon 2 :
    - Chargement du modèle Haar Cascade
    - Détection et sélection du visage principal (surface maximale)
    - Découpage sécurisé du visage avec padding
    - Intégration directe Jalon 2 (Crop) -> Jalon 1 (VGG-16 Embedding)
    """

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print(" [TEST JALON 2] Initialisation du FaceDetector...")
        print("=" * 70)
        cls.detector = FaceDetector()

    def test_01_classifier_initialization(self):
        """Vérifie que le classificateur Haar Cascade est chargé et valide."""
        self.assertIsNotNone(self.detector.classifier)
        self.assertFalse(self.detector.classifier.empty())
        print("  [OK] Test 01 : Classificateur Haar Cascade chargé avec succès.")

    def test_02_get_main_face_selection(self):
        """Vérifie la sélection du visage de surface maximale (w * h)."""
        # 3 visages fictifs : aires = 50*50=2500, 120*100=12000, 70*80=5600
        faces = [
            (10, 20, 50, 50),
            (100, 150, 120, 100),
            (300, 200, 70, 80)
        ]
        main_face = self.detector.get_main_face(faces)
        self.assertEqual(main_face, (100, 150, 120, 100))

        # Test liste vide
        self.assertIsNone(self.detector.get_main_face([]))
        print("  [OK] Test 02 : Sélection exacte du visage principal (aire maximale).")

    def test_03_crop_face_with_padding(self):
        """Vérifie le découpage du visage avec marge et respect des limites de l'image."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (200, 200, 100, 100)  # Face centrale 100x100

        # Avec padding de 10% (0.1) : marge = 10 px de chaque côté -> taille attendue 120x120
        cropped = self.detector.crop_face(frame, bbox, padding_ratio=0.1)
        self.assertEqual(cropped.shape, (120, 120, 3))

        # Cas limite au bord supérieur gauche (0, 0)
        edge_bbox = (5, 5, 50, 50)
        cropped_edge = self.detector.crop_face(frame, edge_bbox, padding_ratio=0.2)
        # Ne doit pas lever d'erreur IndexOutOfBounds
        self.assertGreater(cropped_edge.shape[0], 0)
        self.assertGreater(cropped_edge.shape[1], 0)
        print("  [OK] Test 03 : Découpage sécurisé avec padding conforme.")

    def test_04_integration_crop_to_vgg16(self):
        """
        Vérifie la liaison Jalon 2 -> Jalon 1 :
        Le crop issu du détecteur est directement envoyé à l'extracteur VGG-16
        et produit un embedding 4096-D valide.
        """
        # Simulation d'un visage détecté
        synthetic_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        simulated_face_bbox = (150, 100, 120, 140)

        # 1. Isolation Jalon 2
        face_crop = self.detector.crop_face(synthetic_frame, simulated_face_bbox)

        # 2. Extraction Jalon 1
        extractor = VGG16FeatureExtractor(layer_name="fc1")
        embedding = extractor.extract_features(face_crop, normalize=True)

        self.assertEqual(len(embedding), 4096)
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
        print("  [OK] Test 04 : Pipeline d'intégration [Détection/Crop -> Embedding VGG-16] validé.")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMilestone2FaceDetector)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print(" [RÉSULTAT] TOUS LES TESTS DU JALON 2 ONT RÉUSSI AVEC SUCCÈS !")
        print("=" * 70 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print(" [RÉSULTAT] ÉCHEC DE CERTAINS TESTS DU JALON 2.")
        print("=" * 70 + "\n")
        sys.exit(1)
