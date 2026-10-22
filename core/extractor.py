"""
Module d'extraction de caractéristiques faciales basé sur VGG-16.
Jalon 1 du projet FaceGuard.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image


class VGG16FeatureExtractor:
    """
    Extracteur de vecteurs d'embeddings basé sur le modèle VGG-16 pré-entraîné
    sur ImageNet (Option B : extraction sur la couche dense 'fc1' de dimension 4096).
    """

    TARGET_SIZE = (224, 224)
    EMBEDDING_DIM = 4096

    def __init__(self, layer_name: str = "fc1"):
        """
        Initialise le modèle VGG-16 tronqué jusqu'à la couche dense spécifiée.
        """
        self.layer_name = layer_name
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """
        Charge VGG-16 complet et tronque le modèle pour s'arrêter à la couche 'fc1'.
        """
        print(f"[VGG16FeatureExtractor] Chargement du modèle VGG-16 (couche '{self.layer_name}')...")
        base_vgg = VGG16(weights="imagenet", include_top=True)
        feature_layer = base_vgg.get_layer(self.layer_name).output
        model = Model(inputs=base_vgg.input, outputs=feature_layer, name=f"VGG16_{self.layer_name}")
        model.trainable = False
        print(f"[VGG16FeatureExtractor] Modèle prêt. Dimension de sortie : {self.EMBEDDING_DIM}")
        return model

    def preprocess(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        is_bgr: bool = True
    ) -> np.ndarray:
        """
        Prétraitement strict d'une image pour VGG-16 :
        1. Chargement si chemin fourni.
        2. Conversion de l'espace couleur en RGB.
        3. Redimensionnement en 224x224 (interpolation cv2.INTER_AREA ou INTER_LINEAR).
        4. Ajout de la dimension batch : (1, 224, 224, 3).
        5. Application de preprocess_input de VGG-16 (centrage ImageNet BGR/RGB).

        :param image_input: Chemin du fichier, tableau NumPy (image cv2 ou crop de visage) ou PIL Image.
        :param is_bgr: True si l'image numpy provient d'OpenCV (format BGR standard).
        :return: Tenseur prêt pour l'inférence de forme (1, 224, 224, 3).
        """
        # 1. Chargement de l'image si chemin
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Impossible de charger l'image depuis le chemin : {path_str}")
            is_bgr = True
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("RGB"))
            is_bgr = False
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise TypeError(f"Type d'image non supporté : {type(image_input)}")

        if img.size == 0:
            raise ValueError("L'image fournie est vide.")

        # 2. Conversion en RGB
        if is_bgr:
            if len(img.shape) == 2:  # Niveaux de gris
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # BGRA
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img

        # 3. Redimensionnement en 224x224
        h, w = img_rgb.shape[:2]
        interpolation = cv2.INTER_AREA if (w > self.TARGET_SIZE[0] and h > self.TARGET_SIZE[1]) else cv2.INTER_LINEAR
        img_resized = cv2.resize(img_rgb, self.TARGET_SIZE, interpolation=interpolation)

        # 4. Conversion en float32 et expansion batch
        img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)

        # 5. Normalisation ImageNet VGG-16 (soustraction des moyennes ImageNet)
        # tf.keras.applications.vgg16.preprocess_input attend des images RGB et effectue
        # la conversion BGR + centrage par canal.
        tensor = preprocess_input(img_array)
        return tensor

    def extract_features(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        normalize: bool = True,
        is_bgr: bool = True
    ) -> np.ndarray:
        """
        Extrait le vecteur de caractéristiques VGG-16 (4096-D).

        :param image_input: Image ou chemin de l'image.
        :param normalize: Si True, applique la normalisation L2 pour que la norme soit égale à 1.
        :param is_bgr: True si image en entrée au format BGR OpenCV.
        :return: Vecteur 1D numpy de dimension (4096,).
        """
        tensor = self.preprocess(image_input, is_bgr=is_bgr)

        # Inférence rapide directe (plus légère que predict() pour une seule image)
        features = self.model(tensor, training=False).numpy()
        vector = features.flatten()

        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 1e-12:
                vector = vector / norm
            else:
                vector = np.zeros_like(vector)

        return vector

    @staticmethod
    def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs d'embeddings.
        Retourne une valeur flottante entre -1.0 et 1.0 (typiquement >= 0 pour des visages).
        """
        v1 = np.asarray(vec1, dtype=np.float32).flatten()
        v2 = np.asarray(vec2, dtype=np.float32).flatten()

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0

        similarity = float(np.dot(v1, v2) / (norm1 * norm2))
        return float(np.clip(similarity, -1.0, 1.0))
