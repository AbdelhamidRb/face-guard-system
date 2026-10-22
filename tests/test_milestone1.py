"""
Script de test unitaire et de validation pour le Jalon 1 : Moteur d'extraction VGG-16.
"""

import sys
import os
import unittest
import numpy as np
import cv2

# Ajout de la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.extractor import VGG16FeatureExtractor


class TestMilestone1VGG16(unittest.TestCase):
    """
    Validation unitaire des spécifications du Jalon 1 :
    - Redimensionnement 224x224 et preprocessing VGG-16
    - Extraction d'un vecteur d'embedding 4096-D (couche dense 'fc1')
    - Normalisation et calcul de similarité cosinus
    """

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print(" [TEST JALON 1] Initialisation du VGG16FeatureExtractor...")
        print("=" * 70)
        cls.extractor = VGG16FeatureExtractor(layer_name="fc1")

    def test_01_preprocessing_shape(self):
        """Vérifie que l'image prétraitée a bien la forme (1, 224, 224, 3)."""
        # Image arbitraire (ex: résolution webcam 480x640x3)
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        preprocessed = self.extractor.preprocess(dummy_frame, is_bgr=True)

        self.assertEqual(preprocessed.shape, (1, 224, 224, 3))
        self.assertEqual(preprocessed.dtype, np.float32)
        print("  [OK] Test 01 : Forme du tenseur prétraité = (1, 224, 224, 3)")

    def test_02_feature_extraction_dimension(self):
        """Vérifie que la sortie est un vecteur 1D de 4096 dimensions (Option B)."""
        dummy_frame = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        vector = self.extractor.extract_features(dummy_frame, normalize=False)

        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.ndim, 1)
        self.assertEqual(len(vector), 4096)
        print("  [OK] Test 02 : Dimension du vecteur VGG-16 = 4096")

    def test_03_l2_normalization(self):
        """Vérifie que la normalisation L2 donne une norme euclidienne égale à 1.0."""
        dummy_frame = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)
        vector_normalized = self.extractor.extract_features(dummy_frame, normalize=True)

        norm = np.linalg.norm(vector_normalized)
        self.assertAlmostEqual(norm, 1.0, places=5)
        print("  [OK] Test 03 : Norme L2 du vecteur normalisé = 1.00000")

    def test_04_cosine_similarity_identity(self):
        """Vérifie que la similarité cosinus d'un vecteur avec lui-même vaut 1.0."""
        dummy_frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        vec = self.extractor.extract_features(dummy_frame, normalize=True)

        similarity = self.extractor.compute_similarity(vec, vec)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        print("  [OK] Test 04 : Auto-similarité cosinus = 1.00000")

    def test_05_image_file_loading_and_comparison(self):
        """Vérifie le chargement depuis un fichier image physique et compare 2 images différentes."""
        temp_img1_path = "tests/temp_sample1.jpg"
        temp_img2_path = "tests/temp_sample2.jpg"

        os.makedirs("tests", exist_ok=True)

        # Création de 2 images distinctes (l'une sombre, l'autre claire avec un cercle)
        img1 = np.ones((250, 250, 3), dtype=np.uint8) * 30
        img2 = np.ones((250, 250, 3), dtype=np.uint8) * 220
        cv2.circle(img2, (125, 125), 50, (0, 0, 255), -1)

        cv2.imwrite(temp_img1_path, img1)
        cv2.imwrite(temp_img2_path, img2)

        try:
            vec1 = self.extractor.extract_features(temp_img1_path, normalize=True)
            vec2 = self.extractor.extract_features(temp_img2_path, normalize=True)

            self.assertEqual(len(vec1), 4096)
            self.assertEqual(len(vec2), 4096)

            similarity = self.extractor.compute_similarity(vec1, vec2)
            print(f"  [OK] Test 05 : Extraction depuis fichiers disque réussie.")
            print(f"                 Similarité cosinus entre images distinctes = {similarity:.4f} (< 1.0)")
            self.assertLess(similarity, 0.99)
        finally:
            if os.path.exists(temp_img1_path):
                os.remove(temp_img1_path)
            if os.path.exists(temp_img2_path):
                os.remove(temp_img2_path)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMilestone1VGG16)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print(" [RÉSULTAT] TOUS LES TESTS DU JALON 1 ONT RÉUSSI AVEC SUCCÈS !")
        print("=" * 70 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print(" [RÉSULTAT] ÉCHEC DE CERTAINS TESTS DU JALON 1.")
        print("=" * 70 + "\n")
        sys.exit(1)
