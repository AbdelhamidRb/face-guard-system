"""
Script de test unitaire et de validation pour le Jalon 3 : Enrôlement, base JSON et matching.
"""

import sys
import os
import unittest
from pathlib import Path
import numpy as np
import cv2

# Ajout de la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.database import UserDatabase
from core.matcher import FaceMatcher, AccessDecision
from core.extractor import VGG16FeatureExtractor
from core.detector import FaceDetector
from core.enrollment import EnrollmentManager


class TestMilestone3EnrollmentAndMatching(unittest.TestCase):
    """
    Validation unitaire et fonctionnelle du Jalon 3 :
    - Stockage persistant dans le fichier JSON
    - Algorithme de comparaison biométrique (similarité cosinus & seuil)
    - Statuts d'accès "Autorisée" vs "Refusée"
    - Système d'enrôlement (webcam et fichier image)
    """

    TEST_DB_PATH = Path("tests/test_users.json")

    def setUp(self):
        if self.TEST_DB_PATH.exists():
            self.TEST_DB_PATH.unlink()
        self.db = UserDatabase(db_path=self.TEST_DB_PATH)

    def tearDown(self):
        if self.TEST_DB_PATH.exists():
            self.TEST_DB_PATH.unlink()

    def test_01_database_crud_and_json_persistence(self):
        """Vérifie l'écriture, la lecture et la persistance JSON d'un utilisateur."""
        # Création d'un embedding unitaire 4096-D
        dummy_embedding = np.random.randn(4096).astype(np.float32)
        dummy_embedding /= np.linalg.norm(dummy_embedding)

        # 1. Ajout de l'utilisateur
        user = self.db.add_user(name="Alice Martin", embedding=dummy_embedding)
        self.assertEqual(user["name"], "Alice Martin")
        self.assertEqual(len(user["embedding"]), 4096)
        self.assertTrue(self.TEST_DB_PATH.exists())

        # 2. Rechargement depuis le disque (vérification de la persistance réelle)
        db_reloaded = UserDatabase(db_path=self.TEST_DB_PATH)
        self.assertEqual(db_reloaded.count(), 1)
        self.assertTrue(db_reloaded.user_exists("Alice Martin"))

        retrieved = db_reloaded.get_user("Alice Martin")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["name"], "Alice Martin")
        self.assertEqual(len(retrieved["embedding"]), 4096)

        # 3. Suppression
        deleted = db_reloaded.delete_user("Alice Martin")
        self.assertTrue(deleted)
        self.assertEqual(db_reloaded.count(), 0)
        print("  [OK] Test 01 : Opérations CRUD et persistance JSON validées.")

    def test_02_matcher_empty_database(self):
        """Vérifie qu'une base vide refuse systématiquement tout accès."""
        matcher = FaceMatcher(database=self.db, threshold=0.65)
        query_vec = np.random.randn(4096).astype(np.float32)

        decision = matcher.match(query_vec)
        self.assertFalse(decision.is_authorized)
        self.assertEqual(decision.status, "Refusée")
        self.assertEqual(decision.matched_name, "Inconnu")
        self.assertEqual(decision.similarity_score, 0.0)
        print("  [OK] Test 02 : Rejet automatique sur base vide (Statut: Refusée).")

    def test_03_matcher_authorized_vs_refused(self):
        """Vérifie le verdict d'accès avec seuil de similarité cosinus."""
        # Vecteur de référence pour Bob (normalisé)
        v_bob = np.random.randn(4096).astype(np.float32)
        v_bob /= np.linalg.norm(v_bob)
        self.db.add_user(name="Bob Smith", embedding=v_bob)

        matcher = FaceMatcher(database=self.db, threshold=0.70)

        # 1. Cas légitime : même personne avec une très légère variation (similarité ~ 0.95)
        noise = np.random.randn(4096).astype(np.float32)
        noise /= np.linalg.norm(noise)
        v_bob_probe = 0.95 * v_bob + 0.05 * noise
        v_bob_probe /= np.linalg.norm(v_bob_probe)

        decision_bob = matcher.match(v_bob_probe)
        self.assertTrue(decision_bob.is_authorized)
        self.assertEqual(decision_bob.status, "Autorisée")
        self.assertEqual(decision_bob.matched_name, "Bob Smith")
        self.assertGreater(decision_bob.similarity_score, 0.85)
        print(f"  [OK] Test 03.1 : Accès légitime -> {decision_bob}")

        # 2. Cas imposteur : personne inconnue (vecteur indépendant)
        v_impostor = np.random.randn(4096).astype(np.float32)
        v_impostor /= np.linalg.norm(v_impostor)

        decision_impostor = matcher.match(v_impostor)
        self.assertFalse(decision_impostor.is_authorized)
        self.assertEqual(decision_impostor.status, "Refusée")
        self.assertEqual(decision_impostor.matched_name, "Inconnu")
        self.assertLess(decision_impostor.similarity_score, 0.50)
        print(f"  [OK] Test 03.2 : Accès imposteur -> {decision_impostor}")

    def test_04_dynamic_threshold_adjustment(self):
        """Vérifie que la modification dynamique du seuil ajuste bien le verdict."""
        v_ref = np.random.randn(4096).astype(np.float32)
        v_ref /= np.linalg.norm(v_ref)
        self.db.add_user(name="Claire", embedding=v_ref)

        matcher = FaceMatcher(database=self.db, threshold=0.60)

        # Vecteur intermédiaire avec une similarité cosinus contrôlée d'environ 0.70
        noise = np.random.randn(4096).astype(np.float32)
        noise = noise - np.dot(noise, v_ref) * v_ref  # orthogonal
        noise /= np.linalg.norm(noise)
        v_test = 0.70 * v_ref + np.sqrt(1 - 0.70**2) * noise

        # Avec seuil = 0.60 -> score 0.70 >= 0.60 => "Autorisée"
        matcher.set_threshold(0.60)
        decision1 = matcher.match(v_test)
        self.assertTrue(decision1.is_authorized)
        self.assertEqual(decision1.status, "Autorisée")

        # Avec seuil monté à 0.80 -> score 0.70 < 0.80 => "Refusée"
        matcher.set_threshold(0.80)
        decision2 = matcher.match(v_test)
        self.assertFalse(decision2.is_authorized)
        self.assertEqual(decision2.status, "Refusée")
        print("  [OK] Test 04 : Ajustement dynamique du seuil validé (0.60 -> 0.80).")

    def test_05_enrollment_manager_validation(self):
        """Vérifie le fonctionnement du gestionnaire d'enrôlement."""
        detector = FaceDetector()
        extractor = VGG16FeatureExtractor(layer_name="fc1")
        manager = EnrollmentManager(database=self.db, detector=detector, extractor=extractor)

        # 1. Test rejet image sans visage (image noire)
        blank_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        success, msg, user = manager.enroll_from_frame("Test Sans Visage", blank_frame)
        self.assertFalse(success)
        self.assertIn("Aucun visage", msg)
        self.assertIsNone(user)

        # 2. Test nom vide
        success, msg, user = manager.enroll_from_frame("", blank_frame)
        self.assertFalse(success)
        self.assertIn("vide", msg)

        print("  [OK] Test 05 : Gestionnaire d'enrôlement et validations de sécurité validés.")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMilestone3EnrollmentAndMatching)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print(" [RÉSULTAT] TOUS LES TESTS DU JALON 3 ONT RÉUSSI AVEC SUCCÈS !")
        print("=" * 70 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print(" [RÉSULTAT] ÉCHEC DE CERTAINS TESTS DU JALON 3.")
        print("=" * 70 + "\n")
        sys.exit(1)
