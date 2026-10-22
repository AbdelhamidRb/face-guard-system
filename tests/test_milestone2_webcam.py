"""
Script de démonstration et de validation temps réel du flux webcam (Jalon 2).
Exécute la capture vidéo, la détection Haar Cascade, l'isolation du visage principal
et l'affichage en incrustation (Picture-in-Picture) avec compteur de FPS.
"""

import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

# Ajout du répertoire racine au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detector import FaceDetector


def run_webcam_detection(camera_index: int = 0):
    print("=" * 70)
    print(" [FACEGUARD] Démarrage du flux Webcam - Test du Jalon 2")
    print(" Commandes :")
    print("   - 'q' ou 'Echap' : Quitter")
    print("   - 's'            : Sauvegarder la capture du visage isolé")
    print("=" * 70)

    detector = FaceDetector(
        scale_factor=1.15,
        min_neighbors=5,
        min_size=(70, 70),
        padding_ratio=0.15,
        equalize_hist=True
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la webcam (index {camera_index}).")
        print("Vérifiez qu'une caméra est connectée et non utilisée par une autre application.")
        return

    # Configuration de la résolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_count = 0
    fps_start_time = time.time()
    fps_display = 0.0

    snapshot_dir = Path("data/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[AVERTISSEMENT] Impossible de lire l'image depuis la webcam.")
                break

            # Détection et isolation
            face_crop, main_bbox, all_faces = detector.detect_and_isolate_main_face(frame)

            # Calcul des FPS
            fps_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count = 0
                fps_start_time = time.time()

            # 1. Dessiner les visages secondaires (en gris/bleu)
            for bbox in all_faces:
                if bbox != main_bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)
                    cv2.putText(
                        frame, "Secondaire", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
                    )

            # 2. Dessiner le visage principal (en vert vif avec cadre épais)
            if main_bbox is not None:
                x, y, w, h = main_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Principal ({w}x{h})", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

                # 3. Incrustation (PiP) du visage recadré dans le coin supérieur droit
                if face_crop is not None and face_crop.size > 0:
                    pip_size = (120, 120)
                    pip_crop = cv2.resize(face_crop, pip_size, interpolation=cv2.INTER_AREA)

                    # Bordure blanche autour du PiP
                    h_pip, w_pip = pip_size
                    margin_top, margin_right = 10, 10
                    frame_h, frame_w = frame.shape[:2]

                    x_start = frame_w - w_pip - margin_right
                    y_start = margin_top

                    frame[y_start:y_start + h_pip, x_start:x_start + w_pip] = pip_crop
                    cv2.rectangle(
                        frame,
                        (x_start - 2, y_start - 2),
                        (x_start + w_pip + 2, y_start + h_pip + 2),
                        (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, "Visage Isole", (x_start, y_start + h_pip + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                    )

            # Affichage des métriques et statut
            status_text = "Visage detecte" if main_bbox is not None else "Recherche de visage..."
            status_color = (0, 255, 0) if main_bbox is not None else (0, 165, 255)
            cv2.putText(
                frame, f"Statut : {status_text}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2
            )
            cv2.putText(
                frame, f"FPS : {fps_display:.1f}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
            )
            cv2.putText(
                frame, "Touche 's' pour capturer | 'q' pour quitter", (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
            )

            cv2.imshow("FaceGuard - Test Jalon 2 (Webcam)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' ou ESC
                print("[INFO] Arrêt de la détection webcam par l'utilisateur.")
                break
            elif key == ord('s'):
                if face_crop is not None and face_crop.size > 0:
                    filename = snapshot_dir / f"face_{int(time.time())}.jpg"
                    cv2.imwrite(str(filename), face_crop)
                    print(f"[OK] Visage isolé sauvegardé dans : {filename}")
                else:
                    print("[INFO] Aucun visage à sauvegarder.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_webcam_detection(camera_id)
