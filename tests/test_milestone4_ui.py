"""
Script de test unitaire et de fumée (smoke test) pour le Jalon 4 : Interface Utilisateur Tkinter.
"""

import sys
import os
import unittest
import tkinter as tk

# Ajout de la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.app import FaceGuardApp


class TestMilestone4UI(unittest.TestCase):
    """
    Validation de l'assemblage et du chargement de l'interface Tkinter :
    - Initialisation des composants graphiques
    - Intégration complète de la stack (VGG-16 + Haar Cascade + Base JSON + Matcher)
    - Disponibilité des contrôles d'accès et d'enrôlement
    """

    def test_01_app_initialization_and_layout(self):
        """Vérifie que la fenêtre principale et tous les widgets sont instanciés sans exception."""
        root = tk.Tk()
        # Masquer la fenêtre pendant le test pour ne pas perturber l'écran
        root.withdraw()

        try:
            app = FaceGuardApp(root)

            # Vérification de l'existence des widgets clés
            self.assertIsNotNone(app.video_label)
            self.assertIsNotNone(app.banner_frame)
            self.assertIsNotNone(app.threshold_slider)
            self.assertIsNotNone(app.btn_enroll_cam)
            self.assertIsNotNone(app.btn_enroll_img)
            self.assertIsNotNone(app.users_tree)
            self.assertIsNotNone(app.matcher)
            self.assertIsNotNone(app.detector)
            self.assertIsNotNone(app.extractor)
            self.assertIsNotNone(app.db)

            # Vérification du seuil par défaut
            self.assertAlmostEqual(app.threshold_slider.get(), 0.65, places=2)

            # Forcer une mise à jour de l'arbre graphique Tkinter
            root.update()
            print("  [OK] Test 01 : Instanciation et assemblage graphique Tkinter réussis.")
        finally:
            app._on_close()


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMilestone4UI)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print(" [RÉSULTAT] TOUS LES TESTS DU JALON 4 ONT RÉUSSI AVEC SUCCÈS !")
        print("=" * 70 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print(" [RÉSULTAT] ÉCHEC DU TEST JALON 4.")
        print("=" * 70 + "\n")
        sys.exit(1)
