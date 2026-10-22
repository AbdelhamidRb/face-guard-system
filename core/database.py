"""
Module de gestion de la base de données locale JSON des utilisateurs autorisés.
Jalon 3 du projet FaceGuard.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class UserDatabase:
    """
    Gestionnaire de persistance locale basé sur un fichier JSON pour stocker
    les profils des utilisateurs autorisés et leurs embeddings VGG-16 (4096-D).
    """

    DEFAULT_DB_PATH = Path("data/users.json")

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.users: Dict[str, Dict[str, Any]] = {}
        self.load_database()

    def load_database(self) -> None:
        """
        Charge la base de données depuis le fichier JSON.
        Crée un fichier vide si inexistant.
        """
        if not self.db_path.exists():
            self.users = {}
            self.save_database()
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.users = data
                elif isinstance(data, list):
                    # Rétrocompatibilité si stocké sous forme de liste
                    self.users = {u["name"]: u for u in data if "name" in u}
                else:
                    self.users = {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"[UserDatabase] Erreur de lecture ({e}), réinitialisation de la base.")
            self.users = {}

    def save_database(self) -> None:
        """
        Sauvegarde atomique du dictionnaire des utilisateurs dans le fichier JSON.
        """
        temp_path = self.db_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.users, f, indent=2, ensure_ascii=False)
        temp_path.replace(self.db_path)

    def add_user(
        self,
        name: str,
        embedding: Union[np.ndarray, List[float]],
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enregistre ou met à jour un utilisateur dans la base de données.

        :param name: Nom complet / identifiant de l'utilisateur.
        :param embedding: Vecteur d'embedding 4096-D (numpy array ou liste de float).
        :param image_path: Chemin vers la photo de profil du visage enrôlé.
        :return: L'entrée utilisateur enregistrée.
        """
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Le nom de l'utilisateur ne peut pas être vide.")

        # Conversion du vecteur en liste de floats pour la sérialisation JSON
        if isinstance(embedding, np.ndarray):
            emb_list = [float(x) for x in embedding.flatten()]
        else:
            emb_list = [float(x) for x in embedding]

        user_entry = {
            "name": cleaned_name,
            "embedding": emb_list,
            "dimension": len(emb_list),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": str(image_path) if image_path else None
        }

        self.users[cleaned_name] = user_entry
        self.save_database()
        print(f"[UserDatabase] Utilisateur '{cleaned_name}' enregistré avec succès.")
        return user_entry

    def get_user(self, name: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'un utilisateur par son nom."""
        return self.users.get(name.strip())

    def user_exists(self, name: str) -> bool:
        """Vérifie si un utilisateur est déjà présent."""
        return name.strip() in self.users

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Retourne la liste de tous les utilisateurs enregistrés."""
        return list(self.users.values())

    def count(self) -> int:
        """Nombre d'utilisateurs enregistrés."""
        return len(self.users)

    def delete_user(self, name: str) -> bool:
        """
        Supprime un utilisateur de la base et son image associée si existante.
        """
        cleaned_name = name.strip()
        if cleaned_name in self.users:
            user = self.users.pop(cleaned_name)
            self.save_database()
            # Nettoyage de l'image liée si présente
            if user.get("image_path") and os.path.exists(user["image_path"]):
                try:
                    os.remove(user["image_path"])
                except OSError:
                    pass
            print(f"[UserDatabase] Utilisateur '{cleaned_name}' supprimé.")
            return True
        return False

    def clear(self) -> None:
        """Vide complètement la base de données."""
        self.users = {}
        self.save_database()
