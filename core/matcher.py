"""
Module de comparaison biométrique et de décision de contrôle d'accès.
Jalon 3 du projet FaceGuard.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from .database import UserDatabase


@dataclass
class AccessDecision:
    """Structure représentant le verdict de contrôle d'accès."""
    is_authorized: bool
    status: str            # "Autorisée" ou "Refusée"
    matched_name: str      # Nom de l'utilisateur ou "Inconnu"
    similarity_score: float # Score de similarité cosinus (entre 0.0 et 1.0)
    threshold: float       # Seuil appliqué
    user_data: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        symbol = "[ACCÈS AUTORISÉ]" if self.is_authorized else "[ACCÈS REFUSÉ]"
        return f"{symbol} Utilisateur: {self.matched_name} | Score: {self.similarity_score:.4f} (Seuil: {self.threshold:.2f})"


class FaceMatcher:
    """
    Compare un embedding facial scanné en direct avec l'ensemble des signatures
    enregistrées dans la base de données JSON pour statuer sur l'autorisation d'accès.
    """

    DEFAULT_THRESHOLD = 0.65

    def __init__(self, database: UserDatabase, threshold: float = DEFAULT_THRESHOLD):
        self.database = database
        self.threshold = threshold

    def set_threshold(self, new_threshold: float) -> None:
        """Met à jour le seuil de décision (ajustable en direct depuis l'UI Tkinter)."""
        self.threshold = max(0.0, min(1.0, float(new_threshold)))

    def match(self, scanned_embedding: np.ndarray) -> AccessDecision:
        """
        Compare le vecteur scanné avec tous les profils enregistrés dans la base.
        Sélectionne le profil ayant la plus grande similarité cosinus.

        :param scanned_embedding: Vecteur 4096-D issu du Jalon 1 (normalisé L2).
        :return: AccessDecision ("Autorisée" si score >= seuil, sinon "Refusée").
        """
        users = self.database.get_all_users()

        # Si aucun utilisateur dans la base -> Refus systématique
        if not users:
            return AccessDecision(
                is_authorized=False,
                status="Refusée",
                matched_name="Inconnu",
                similarity_score=0.0,
                threshold=self.threshold,
                user_data=None
            )

        scanned_vec = np.asarray(scanned_embedding, dtype=np.float32).flatten()
        scanned_norm = np.linalg.norm(scanned_vec)
        if scanned_norm > 1e-12:
            scanned_vec = scanned_vec / scanned_norm

        best_user = None
        best_similarity = -1.0

        for user in users:
            user_emb = np.asarray(user["embedding"], dtype=np.float32).flatten()
            user_norm = np.linalg.norm(user_emb)
            if user_norm > 1e-12:
                user_emb = user_emb / user_norm

            # Similarité cosinus
            similarity = float(np.dot(scanned_vec, user_emb))

            if similarity > best_similarity:
                best_similarity = similarity
                best_user = user

        # Écrêtage entre 0.0 et 1.0
        best_similarity = float(np.clip(best_similarity, 0.0, 1.0))

        # Décision selon le seuil de similarité
        is_authorized = best_similarity >= self.threshold
        status = "Autorisée" if is_authorized else "Refusée"
        matched_name = best_user["name"] if is_authorized else "Inconnu"

        return AccessDecision(
            is_authorized=is_authorized,
            status=status,
            matched_name=matched_name,
            similarity_score=best_similarity,
            threshold=self.threshold,
            user_data=best_user if is_authorized else None
        )

    def match_all_scores(self, scanned_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """
        Retourne la liste triée de tous les utilisateurs et de leurs scores respectifs.
        """
        users = self.database.get_all_users()
        if not users:
            return []

        scanned_vec = np.asarray(scanned_embedding, dtype=np.float32).flatten()
        scanned_norm = np.linalg.norm(scanned_vec)
        if scanned_norm > 1e-12:
            scanned_vec = scanned_vec / scanned_norm

        scores = []
        for user in users:
            user_emb = np.asarray(user["embedding"], dtype=np.float32).flatten()
            user_norm = np.linalg.norm(user_emb)
            if user_norm > 1e-12:
                user_emb = user_emb / user_norm

            sim = float(np.clip(np.dot(scanned_vec, user_emb), 0.0, 1.0))
            scores.append((user["name"], sim))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores
