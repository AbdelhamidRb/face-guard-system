"""
Package core pour le système de reconnaissance faciale FaceGuard.
"""

from .extractor import VGG16FeatureExtractor
from .detector import FaceDetector
from .database import UserDatabase
from .matcher import FaceMatcher, AccessDecision
from .enrollment import EnrollmentManager

__all__ = [
    "VGG16FeatureExtractor",
    "FaceDetector",
    "UserDatabase",
    "FaceMatcher",
    "AccessDecision",
    "EnrollmentManager",
]
