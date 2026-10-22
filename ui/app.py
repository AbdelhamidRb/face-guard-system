"""
Application graphique Tkinter pour le Système de Contrôle d'Accès par Reconnaissance Faciale (VGG-16).
Jalon 4 du projet FaceGuard.
"""

from datetime import datetime
import os
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageTk

from core.database import UserDatabase
from core.detector import FaceDetector
from core.enrollment import EnrollmentManager
from core.extractor import VGG16FeatureExtractor
from core.matcher import FaceMatcher, AccessDecision


class FaceGuardApp:
    """
    Interface graphique principale FaceGuard sous Tkinter :
    - Flux vidéo webcam OpenCV temps réel avec incrustation et bounding boxes
    - Bannière d'accès visuel dynamique (Vert: Autorisée / Rouge: Refusée / Gris: En attente)
    - Curseur de réglage en direct du seuil de similarité cosinus
    - Système double d'enrôlement (Capture directe Webcam & Importation d'image)
    - Gestion de la liste des utilisateurs autorisés (visualisation & suppression)
    """

    COLOR_BG = "#1e1e2e"
    COLOR_CARD = "#252538"
    COLOR_TEXT = "#f8f9fa"
    COLOR_MUTED = "#a0a0b0"
    COLOR_AUTHORIZED = "#2ecc71"  # Vert vif
    COLOR_REFUSED = "#e74c3c"     # Rouge vif
    COLOR_WAITING = "#6c757d"     # Gris neutre
    COLOR_ACCENT = "#3498db"      # Bleu action

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FaceGuard - Contrôle d'Accès par Reconnaissance Faciale (VGG-16)")
        self.root.geometry("1180x760")
        self.root.minsize(1050, 700)
        self.root.configure(bg=self.COLOR_BG)

        # 1. Initialisation des composants Core
        self.db = UserDatabase()
        self.detector = FaceDetector(
            scale_factor=1.15,
            min_neighbors=5,
            min_size=(70, 70),
            padding_ratio=0.15,
            equalize_hist=True
        )
        self.extractor = VGG16FeatureExtractor(layer_name="fc1")
        self.matcher = FaceMatcher(database=self.db, threshold=0.65)
        self.enrollment_manager = EnrollmentManager(
            database=self.db,
            detector=self.detector,
            extractor=self.extractor
        )

        # 2. Variables d'état
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.latest_face_crop: Optional[np.ndarray] = None

        # Threading pour la reconnaissance VGG-16 sans ralentir le rendu vidéo
        self.last_decision: Optional[AccessDecision] = None
        self.is_recognizing = False
        self.recognition_interval = 0.25  # Inférence toutes les 250 ms maximum
        self.last_recognition_time = 0.0

        # FPS counter
        self.fps_count = 0
        self.fps_start = time.time()
        self.current_fps = 0.0

        # 3. Construction des éléments graphiques
        self._setup_styles()
        self._create_widgets()
        self._refresh_users_list()

        # 4. Lancement de la caméra
        self._start_camera()

        # Protocole de fermeture propre
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=self.COLOR_BG, foreground=self.COLOR_TEXT)
        style.configure("Card.TFrame", background=self.COLOR_CARD)
        style.configure("Treeview",
                        background="#181825",
                        foreground=self.COLOR_TEXT,
                        fieldbackground="#181825",
                        rowheight=26,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading",
                        background=self.COLOR_CARD,
                        foreground=self.COLOR_TEXT,
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", self.COLOR_ACCENT)])

    def _create_widgets(self):
        # ----------------------------------------------------
        # EN-TÊTE : Titre et Bannière Visuelle de Contrôle d'Accès
        # ----------------------------------------------------
        header_frame = tk.Frame(self.root, bg=self.COLOR_CARD, height=85)
        header_frame.pack(fill=tk.X, side=tk.TOP, padx=12, pady=(10, 8))
        header_frame.pack_propagate(False)

        title_sub_frame = tk.Frame(header_frame, bg=self.COLOR_CARD)
        title_sub_frame.pack(side=tk.LEFT, padx=15, fill=tk.Y)

        title_lbl = tk.Label(
            title_sub_frame,
            text="🛡️ FaceGuard Access Control",
            font=("Segoe UI", 16, "bold"),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_CARD
        )
        title_lbl.pack(anchor="w", pady=(8, 0))

        subtitle_lbl = tk.Label(
            title_sub_frame,
            text="VGG-16 Deep Biometric Identification (4096-D) • Haar Cascade Real-Time Tracking",
            font=("Segoe UI", 9),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        )
        subtitle_lbl.pack(anchor="w")

        # Bannière d'accès dynamique (Vert / Rouge / Gris)
        self.banner_frame = tk.Frame(header_frame, bg=self.COLOR_WAITING, padx=20, pady=6)
        self.banner_frame.pack(side=tk.RIGHT, padx=15, pady=10, fill=tk.Y)

        self.banner_icon_lbl = tk.Label(
            self.banner_frame,
            text="⚪",
            font=("Segoe UI", 18),
            fg="white",
            bg=self.COLOR_WAITING
        )
        self.banner_icon_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self.banner_text_lbl = tk.Label(
            self.banner_frame,
            text="EN ATTENTE DE VISAGE",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg=self.COLOR_WAITING
        )
        self.banner_text_lbl.pack(side=tk.LEFT)

        self.banner_subtext_lbl = tk.Label(
            self.banner_frame,
            text="Placez-vous devant la caméra",
            font=("Segoe UI", 9),
            fg="white",
            bg=self.COLOR_WAITING
        )
        self.banner_subtext_lbl.pack(side=tk.LEFT, padx=(10, 0))

        # ----------------------------------------------------
        # CORPS DE L'INTERFACE : Vidéo (Gauche) + Contrôles (Droite)
        # ----------------------------------------------------
        main_content = tk.Frame(self.root, bg=self.COLOR_BG)
        main_content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        # ZONE VIDÉO (GAUCHE)
        video_card = tk.Frame(main_content, bg=self.COLOR_CARD, bd=1, relief=tk.SOLID)
        video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        video_header = tk.Frame(video_card, bg=self.COLOR_CARD)
        video_header.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(
            video_header,
            text="🎥 Flux Vidéo en Direct",
            font=("Segoe UI", 11, "bold"),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_CARD
        ).pack(side=tk.LEFT)

        self.fps_lbl = tk.Label(
            video_header,
            text="FPS: 0.0",
            font=("Consolas", 10, "bold"),
            fg="#f1c40f",
            bg=self.COLOR_CARD
        )
        self.fps_lbl.pack(side=tk.RIGHT)

        # Conteneur d'image vidéo
        self.video_label = tk.Label(video_card, bg="#0d0e15")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # PANNEAU LATÉRAL (DROITE)
        control_panel = tk.Frame(main_content, bg=self.COLOR_CARD, width=420)
        control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        control_panel.pack_propagate(False)

        # 1. Section Réglage du Seuil (Slider)
        threshold_frame = tk.LabelFrame(
            control_panel,
            text="⚙️ Sensibilité de Décision (Seuil de Similarité)",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_CARD,
            padx=10,
            pady=8
        )
        threshold_frame.pack(fill=tk.X, padx=10, pady=(10, 8))

        thresh_info = tk.Frame(threshold_frame, bg=self.COLOR_CARD)
        thresh_info.pack(fill=tk.X)
        tk.Label(thresh_info, text="Seuil Cosinus :", font=("Segoe UI", 9), fg=self.COLOR_MUTED, bg=self.COLOR_CARD).pack(side=tk.LEFT)
        self.threshold_value_lbl = tk.Label(
            thresh_info,
            text=f"{self.matcher.threshold:.2f}",
            font=("Segoe UI", 10, "bold"),
            fg=self.COLOR_ACCENT,
            bg=self.COLOR_CARD
        )
        self.threshold_value_lbl.pack(side=tk.RIGHT)

        self.threshold_slider = tk.Scale(
            threshold_frame,
            from_=0.40,
            to=0.95,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg=self.COLOR_CARD,
            troughcolor="#181825",
            activebackground=self.COLOR_ACCENT,
            highlightthickness=0,
            command=self._on_threshold_change
        )
        self.threshold_slider.set(self.matcher.threshold)
        self.threshold_slider.pack(fill=tk.X, pady=(5, 0))

        # 2. Section Enrôlement Double
        enroll_frame = tk.LabelFrame(
            control_panel,
            text="👤 Enrôlement d'un Nouvel Utilisateur",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_CARD,
            padx=10,
            pady=8
        )
        enroll_frame.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(
            enroll_frame,
            text="Nom & Prénom :",
            font=("Segoe UI", 9),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        ).pack(anchor="w")

        self.name_entry = tk.Entry(
            enroll_frame,
            font=("Segoe UI", 10),
            bg="#181825",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT
        )
        self.name_entry.pack(fill=tk.X, pady=(3, 8), ipady=3)

        btn_box = tk.Frame(enroll_frame, bg=self.COLOR_CARD)
        btn_box.pack(fill=tk.X, pady=2)

        self.btn_enroll_cam = tk.Button(
            btn_box,
            text="📷 Capturer Webcam",
            font=("Segoe UI", 9, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#2ecc71",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._enroll_from_webcam
        )
        self.btn_enroll_cam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=4)

        self.btn_enroll_img = tk.Button(
            btn_box,
            text="📁 Importer Image",
            font=("Segoe UI", 9, "bold"),
            bg=self.COLOR_ACCENT,
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._enroll_from_file
        )
        self.btn_enroll_img.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0), ipady=4)

        self.enroll_status_lbl = tk.Label(
            enroll_frame,
            text="",
            font=("Segoe UI", 8),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD,
            wraplength=380,
            justify=tk.LEFT
        )
        self.enroll_status_lbl.pack(fill=tk.X, pady=(6, 0))

        # 3. Section Liste des Utilisateurs Autorisés
        users_frame = tk.LabelFrame(
            control_panel,
            text="📋 Utilisateurs Autorisés Enregistrés",
            font=("Segoe UI", 9, "bold"),
            fg=self.COLOR_TEXT,
            bg=self.COLOR_CARD,
            padx=10,
            pady=8
        )
        users_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        tree_scroll = ttk.Scrollbar(users_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.users_tree = ttk.Treeview(
            users_frame,
            columns=("name", "created_at"),
            show="headings",
            selectmode="browse",
            yscrollcommand=tree_scroll.set
        )
        tree_scroll.config(command=self.users_tree.yview)

        self.users_tree.heading("name", text="Nom Utilisateur")
        self.users_tree.heading("created_at", text="Date d'Enrôlement")
        self.users_tree.column("name", width=190, anchor="w")
        self.users_tree.column("created_at", width=140, anchor="center")
        self.users_tree.pack(fill=tk.BOTH, expand=True)

        users_action_box = tk.Frame(users_frame, bg=self.COLOR_CARD)
        users_action_box.pack(fill=tk.X, pady=(8, 0))

        self.user_count_lbl = tk.Label(
            users_action_box,
            text="0 utilisateur(s)",
            font=("Segoe UI", 9),
            fg=self.COLOR_MUTED,
            bg=self.COLOR_CARD
        )
        self.user_count_lbl.pack(side=tk.LEFT)

        self.btn_delete_user = tk.Button(
            users_action_box,
            text="🗑️ Supprimer",
            font=("Segoe UI", 8, "bold"),
            bg="#c0392b",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._delete_selected_user
        )
        self.btn_delete_user.pack(side=tk.RIGHT, padx=(4, 0))

        self.btn_refresh = tk.Button(
            users_action_box,
            text="🔄 Actualiser",
            font=("Segoe UI", 8),
            bg="#34495e",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._refresh_users_list
        )
        self.btn_refresh.pack(side=tk.RIGHT, padx=(0, 4))

        # 4. Pied de page du panneau latéral
        footer_frame = tk.Frame(control_panel, bg=self.COLOR_CARD)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=8)

        self.btn_toggle_cam = tk.Button(
            footer_frame,
            text="⏹️ Arrêter Caméra",
            font=("Segoe UI", 9, "bold"),
            bg="#7f8c8d",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._toggle_camera
        )
        self.btn_toggle_cam.pack(fill=tk.X, ipady=3)

    # ----------------------------------------------------
    # GESTION VIDÉO & RECONNAISSANCE EN TEMPS RÉEL
    # ----------------------------------------------------
    def _start_camera(self):
        """Initialise et lance la capture webcam."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            self._update_banner(
                status="ERROR",
                title="CAMÉRA INDISPONIBLE",
                subtitle="Impossible d'accéder au périphérique vidéo.",
                color=self.COLOR_REFUSED
            )
            return

        self.is_camera_running = True
        self.btn_toggle_cam.config(text="⏹️ Arrêter Caméra", bg="#7f8c8d")
        self._video_loop()

    def _toggle_camera(self):
        """Met en pause ou redémarre la webcam."""
        if self.is_camera_running:
            self.is_camera_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.btn_toggle_cam.config(text="▶️ Démarrer Caméra", bg=self.COLOR_ACCENT)
            self.video_label.config(image="")
            self._update_banner(
                status="IDLE",
                title="CAMÉRA EN PAUSE",
                subtitle="Cliquez sur Démarrer pour reprendre",
                color=self.COLOR_WAITING
            )
        else:
            self._start_camera()

    def _video_loop(self):
        """Boucle d'affichage vidéo synchronisée avec Tkinter."""
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_frame = frame.copy()

            # 1. Détection du visage avec Haar Cascades
            face_crop, main_bbox, all_faces = self.detector.detect_and_isolate_main_face(frame)
            self.latest_face_crop = face_crop

            # 2. Reconnaissance asynchrone VGG-16 pour fluidité optimale
            now = time.time()
            if face_crop is not None:
                if not self.is_recognizing and (now - self.last_recognition_time) >= self.recognition_interval:
                    self.last_recognition_time = now
                    # Lancement de l'extraction VGG-16 sur un thread secondaire
                    threading.Thread(
                        target=self._run_async_recognition,
                        args=(face_crop,),
                        daemon=True
                    ).start()
            else:
                self.last_decision = None
                self._update_banner(
                    status="WAITING",
                    title="EN ATTENTE DE VISAGE",
                    subtitle="Aucun visage détecté devant la caméra",
                    color=self.COLOR_WAITING
                )

            # 3. Dessiner les annotations sur l'image
            display_frame = frame.copy()

            # Visages secondaires (gris)
            for bbox in all_faces:
                if bbox != main_bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (120, 120, 120), 1)

            # Visage principal
            if main_bbox is not None:
                x, y, w, h = main_bbox

                # Couleur du cadre selon le statut courant
                if self.last_decision and self.last_decision.is_authorized:
                    box_color = (46, 204, 113)  # Vert BGR
                    label_text = f"AUTORISE : {self.last_decision.matched_name} ({self.last_decision.similarity_score:.0%})"
                elif self.last_decision and not self.last_decision.is_authorized:
                    box_color = (60, 76, 231)   # Rouge BGR
                    label_text = f"REFUSE (Score: {self.last_decision.similarity_score:.0%})"
                else:
                    box_color = (255, 165, 0)   # Orange / Analyse
                    label_text = "Analyse VGG-16..."

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.rectangle(display_frame, (x, max(0, y - 24)), (x + w, y), box_color, -1)
                cv2.putText(
                    display_frame, label_text, (x + 5, max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )

                # Incrustation miniature (PiP) du visage extrait
                if face_crop is not None and face_crop.size > 0:
                    try:
                        pip_crop = cv2.resize(face_crop, (100, 100), interpolation=cv2.INTER_AREA)
                        fh, fw = display_frame.shape[:2]
                        display_frame[10:110, fw - 110:fw - 10] = pip_crop
                        cv2.rectangle(display_frame, (fw - 112, 8), (fw - 8, 112), (255, 255, 255), 1)
                    except Exception:
                        pass

            # 4. Calcul FPS
            self.fps_count += 1
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                self.current_fps = self.fps_count / elapsed
                self.fps_count = 0
                self.fps_start = time.time()
                self.fps_lbl.config(text=f"FPS: {self.current_fps:.1f}")

            # 5. Conversion BGR -> RGB -> PhotoImage Tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            # Redimensionnement adapté au conteneur Tkinter
            lbl_w = max(100, self.video_label.winfo_width())
            lbl_h = max(100, self.video_label.winfo_height())
            if lbl_w > 100 and lbl_h > 100:
                img.thumbnail((lbl_w, lbl_h), Image.Resampling.BILINEAR)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Planifier la prochaine trame (environ 30 FPS = ~33 ms)
        self.root.after(25, self._video_loop)

    def _run_async_recognition(self, face_crop: np.ndarray):
        """Exécution de l'inférence VGG-16 et matching en tâche de fond."""
        self.is_recognizing = True
        try:
            embedding = self.extractor.extract_features(face_crop, normalize=True, is_bgr=True)
            decision = self.matcher.match(embedding)
            self.last_decision = decision

            # Mise à jour de la bannière sur le thread principal
            self.root.after(0, self._apply_decision_ui, decision)
        except Exception as e:
            print(f"[FaceGuardApp] Erreur de reconnaissance : {e}")
        finally:
            self.is_recognizing = False

    def _apply_decision_ui(self, decision: AccessDecision):
        """Met à jour l'affichage de la bannière selon le verdict d'accès."""
        if decision.is_authorized:
            self._update_banner(
                status="AUTHORIZED",
                title=f"ACCÈS AUTORISÉ : {decision.matched_name}",
                subtitle=f"Score de similarité : {decision.similarity_score:.1%} (Seuil : {decision.threshold:.2f})",
                color=self.COLOR_AUTHORIZED
            )
        else:
            self._update_banner(
                status="REFUSED",
                title="ACCÈS REFUSÉ : PERSONNE INCONNUE",
                subtitle=f"Score insuffisant : {decision.similarity_score:.1%} (Seuil requis : {decision.threshold:.2f})",
                color=self.COLOR_REFUSED
            )

    def _update_banner(self, status: str, title: str, subtitle: str, color: str):
        """Ajuste les couleurs et textes de la bannière supérieure."""
        icons = {
            "AUTHORIZED": "✔",
            "REFUSED": "✖",
            "WAITING": "⚪",
            "ERROR": "⚠️",
            "IDLE": "⏸️"
        }
        self.banner_frame.configure(bg=color)
        self.banner_icon_lbl.configure(text=icons.get(status, "⚪"), bg=color)
        self.banner_text_lbl.configure(text=title, bg=color)
        self.banner_subtext_lbl.configure(text=subtitle, bg=color)

    # ----------------------------------------------------
    # ENRÔLEMENT & GESTION DES PROFILS
    # ----------------------------------------------------
    def _on_threshold_change(self, value):
        val = float(value)
        self.threshold_value_lbl.config(text=f"{val:.2f}")
        self.matcher.set_threshold(val)

    def _enroll_from_webcam(self):
        """Enrôle l'utilisateur à partir du visage actuellement capturé par la webcam."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Nom requis", "Veuillez saisir un nom pour l'utilisateur à enrôler.")
            return

        if self.current_frame is None or self.latest_face_crop is None:
            messagebox.showwarning("Aucun visage", "Aucun visage n'est actuellement visible devant la caméra.")
            return

        # Exécution de l'enrôlement
        success, msg, _ = self.enrollment_manager.enroll_from_frame(name, self.current_frame)
        self._handle_enrollment_result(success, msg)

    def _enroll_from_file(self):
        """Enrôle un utilisateur depuis une image sélectionnée sur le disque."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Nom requis", "Veuillez saisir un nom pour l'utilisateur avant d'importer une photo.")
            return

        file_path = filedialog.askopenfilename(
            title="Sélectionner une photo pour l'enrôlement",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tous les fichiers", "*.*")
            ]
        )
        if not file_path:
            return

        success, msg, _ = self.enrollment_manager.enroll_from_image_path(name, file_path)
        self._handle_enrollment_result(success, msg)

    def _handle_enrollment_result(self, success: bool, message: str):
        if success:
            self.name_entry.delete(0, tk.END)
            self.enroll_status_lbl.config(text=f"✔ {message}", fg=self.COLOR_AUTHORIZED)
            self._refresh_users_list()
            messagebox.showinfo("Enrôlement réussi", message)
        else:
            self.enroll_status_lbl.config(text=f"✖ {message}", fg=self.COLOR_REFUSED)
            messagebox.showerror("Échec de l'enrôlement", message)

    def _refresh_users_list(self):
        """Actualise le tableau des utilisateurs enregistrés."""
        for item in self.users_tree.get_children():
            self.users_tree.delete(item)

        users = self.db.get_all_users()
        for user in sorted(users, key=lambda u: u.get("created_at", ""), reverse=True):
            self.users_tree.insert(
                "",
                tk.END,
                values=(user.get("name", "N/A"), user.get("created_at", "N/A"))
            )

        self.user_count_lbl.config(text=f"{len(users)} utilisateur(s) autorisé(s)")

    def _delete_selected_user(self):
        """Supprime l'utilisateur sélectionné dans le tableau."""
        selected = self.users_tree.selection()
        if not selected:
            messagebox.showinfo("Sélection", "Veuillez sélectionner un utilisateur à supprimer.")
            return

        item = self.users_tree.item(selected[0])
        user_name = item["values"][0]

        confirm = messagebox.askyesno(
            "Confirmation",
            f"Êtes-vous sûr de vouloir supprimer l'utilisateur '{user_name}' de la base d'accès ?"
        )
        if confirm:
            self.db.delete_user(user_name)
            self._refresh_users_list()
            messagebox.showinfo("Suppression", f"L'utilisateur '{user_name}' a été retiré des accès autorisés.")

    def _on_close(self):
        """Fermeture propre de l'application et libération des ressources matérielles."""
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


def run_app():
    root = tk.Tk()
    app = FaceGuardApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()
