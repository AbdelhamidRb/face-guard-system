"""
Module de détection et d'isolation de visages basé sur OpenCV et Haar Cascades.
Jalon 2 du projet FaceGuard.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


class FaceDetector:
    """
    Détecteur de visages utilisant le classificateur Haar Cascade frontal d'OpenCV.
    Permet de repérer les visages et d'extraire le visage principal (surface maximale).
    """

    DEFAULT_CASCADE_PATH = Path("models/haarcascade_frontalface_default.xml")

    def __init__(
        self,
        cascade_path: Optional[Union[str, Path]] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 60),
        padding_ratio: float = 0.15,
        equalize_hist: bool = True
    ):
        """
        Initialise le détecteur Haar Cascade.

        :param cascade_path: Chemin vers le fichier XML Haar Cascade.
        :param scale_factor: Facteur d'échelle pour multiscale (1.1 standard).
        :param min_neighbors: Nombre minimal de voisins pour valider un visage (5 par défaut).
        :param min_size: Taille minimale (largeur, hauteur) d'un visage détecté.
        :param padding_ratio: Pourcentage de marge ajoutée autour du visage recadré (0.15 = +15%).
        :param equalize_hist: Applique une égalisation d'histogramme pour améliorer la robustesse à la lumière.
        """
        self.cascade_path = Path(cascade_path) if cascade_path else self.DEFAULT_CASCADE_PATH
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.padding_ratio = padding_ratio
        self.equalize_hist = equalize_hist

        self.classifier = self._load_classifier()

    def _load_classifier(self) -> cv2.CascadeClassifier:
        """
        Charge et valide le fichier XML Haar Cascade.
        """
        if not self.cascade_path.exists():
            # Fallback éventuel sur cv2.data si disponible
            cv2_builtin = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cv2_builtin.exists():
                self.cascade_path = cv2_builtin
            else:
                raise FileNotFoundError(
                    f"Fichier Haar Cascade introuvable : {self.cascade_path}. "
                    f"Vérifiez que le fichier XML est présent dans le dossier 'models/'."
                )

        classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if classifier.empty():
            raise RuntimeError(f"Échec du chargement du classificateur Haar Cascade depuis {self.cascade_path}")

        return classifier

    def detect_all_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Détecte tous les visages présents dans l'image.

        :param frame: Image OpenCV au format BGR ou niveau de gris.
        :return: Liste de tuples (x, y, w, h) représentant les boîtes englobantes.
        """
        if frame is None or frame.size == 0:
            return []

        # Conversion en niveaux de gris si nécessaire
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Égalisation d'histogramme pour normaliser le contraste
        if self.equalize_hist:
            gray = cv2.equalizeHist(gray)

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return []

        # Conversion en liste de tuples Python standards (x, y, w, h)
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def get_main_face(
        self,
        faces: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Sélectionne le visage principal parmi une liste de visages détectés.
        Règle : Visage ayant la surface (w * h) maximale.

        :param faces: Liste de coordonnées (x, y, w, h).
        :return: La boîte du visage principal ou None si aucun visage.
        """
        if not faces:
            return None

        # Tri par aire décroissante (w * h)
        main_face = max(faces, key=lambda bbox: bbox[2] * bbox[3])
        return main_face

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding_ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        Isole et découpe la région du visage avec une marge de sécurité.
        Les coordonnées sont strictement contraintes aux dimensions de l'image.

        :param frame: Image source NumPy.
        :param bbox: Boîte englobante (x, y, w, h).
        :param padding_ratio: Marge relative (utilise la valeur de l'instance si None).
        :return: Image découpée (crop) du visage.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Image source invalide ou vide.")

        ratio = padding_ratio if padding_ratio is not None else self.padding_ratio
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        # Calcul des marges
        pad_x = int(w * ratio)
        pad_y = int(h * ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_frame, x + w + pad_x)
        y2 = min(h_frame, y + h + pad_y)

        cropped = frame[y1:y2, x1:x2].copy()
        return cropped

    def detect_and_isolate_main_face(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Pipeline complet :
        1. Détecte tous les visages.
        2. Isole le visage principal.
        3. Découpe la région du visage principal.

        :param frame: Image source OpenCV.
        :return: Tuple (visage_découpé_ou_None, bbox_principale_ou_None, liste_tous_les_visages).
        """
        all_faces = self.detect_all_faces(frame)
        if not all_faces:
            return None, None, []

        main_face = self.get_main_face(all_faces)
        face_crop = self.crop_face(frame, main_face)

        return face_crop, main_face, all_faces
