"""
Module d'enrôlement des utilisateurs (système double : capture webcam ou fichier image).
Jalon 3 du projet FaceGuard.
"""

from pathlib import Path
import re
import time
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from .database import UserDatabase
from .detector import FaceDetector
from .extractor import VGG16FeatureExtractor


class EnrollmentManager:
    """
    Orchestre le système double d'enrôlement :
    1. Enrôlement par capture webcam en direct.
    2. Enrôlement par importation d'un fichier image (JPG / PNG).
    """

    ENROLLED_DIR = Path("data/enrolled")

    def __init__(
        self,
        database: UserDatabase,
        detector: FaceDetector,
        extractor: VGG16FeatureExtractor
    ):
        self.database = database
        self.detector = detector
        self.extractor = extractor
        self.ENROLLED_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Nettoie le nom de l'utilisateur pour un nom de fichier sécurisé."""
        clean = re.sub(r"[^\w\s-]", "", name).strip().lower()
        return re.sub(r"[-\s]+", "_", clean)

    def _save_face_crop(self, face_crop: np.ndarray, user_name: str) -> Path:
        """Sauvegarde le crop du visage dans le dossier data/enrolled."""
        safe_name = self._sanitize_filename(user_name)
        timestamp = int(time.time())
        filename = f"{safe_name}_{timestamp}.jpg"
        save_path = self.ENROLLED_DIR / filename
        cv2.imwrite(str(save_path), face_crop)
        return save_path

    def enroll_from_frame(
        self,
        name: str,
        frame: np.ndarray
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Enrôle un utilisateur à partir d'une image/frame issue de la webcam.

        :param name: Nom de l'utilisateur.
        :param frame: Image OpenCV (BGR) de la webcam.
        :return: (Succès, Message, Données utilisateur ou None).
        """
        name_clean = name.strip()
        if not name_clean:
            return False, "Le nom d'utilisateur ne peut pas être vide.", None

        if frame is None or frame.size == 0:
            return False, "La frame capturée est invalide.", None

        # 1. Détection et isolation du visage principal
        face_crop, main_bbox, _ = self.detector.detect_and_isolate_main_face(frame)
        if face_crop is None or main_bbox is None:
            return False, "Aucun visage détecté sur la capture webcam.", None

        # 2. Extraction du vecteur VGG-16 (4096-D normalisé)
        embedding = self.extractor.extract_features(face_crop, normalize=True, is_bgr=True)

        # 3. Sauvegarde de la photo du visage isolé
        saved_img_path = self._save_face_crop(face_crop, name_clean)

        # 4. Enregistrement dans la base JSON
        user_entry = self.database.add_user(
            name=name_clean,
            embedding=embedding,
            image_path=str(saved_img_path)
        )

        return True, f"Utilisateur '{name_clean}' enrôlé avec succès via la webcam.", user_entry

    def enroll_from_image_path(
        self,
        name: str,
        image_path: Union[str, Path]
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Enrôle un utilisateur à partir d'un fichier image physique (PNG/JPG).

        :param name: Nom de l'utilisateur.
        :param image_path: Chemin du fichier image sur le disque.
        :return: (Succès, Message, Données utilisateur ou None).
        """
        name_clean = name.strip()
        if not name_clean:
            return False, "Le nom d'utilisateur ne peut pas être vide.", None

        file_path = Path(image_path)
        if not file_path.exists():
            return False, f"Le fichier image est introuvable : {file_path}", None

        # Lecture de l'image avec OpenCV
        frame = cv2.imread(str(file_path))
        if frame is None:
            return False, f"Impossible de lire le fichier image : {file_path}", None

        # 1. Détection et isolation du visage
        face_crop, main_bbox, _ = self.detector.detect_and_isolate_main_face(frame)
        if face_crop is None or main_bbox is None:
            # Si aucun visage détecté par Haar Cascade, avertissement explicite
            return False, (
                "Aucun visage n'a été détecté dans l'image importée. "
                "Veuillez choisir une photo claire avec le visage bien visible de face."
            ), None

        # 2. Extraction VGG-16
        embedding = self.extractor.extract_features(face_crop, normalize=True, is_bgr=True)

        # 3. Sauvegarde de la vignette
        saved_img_path = self._save_face_crop(face_crop, name_clean)

        # 4. Enregistrement JSON
        user_entry = self.database.add_user(
            name=name_clean,
            embedding=embedding,
            image_path=str(saved_img_path)
        )

        return True, f"Utilisateur '{name_clean}' enrôlé avec succès depuis le fichier image.", user_entry
