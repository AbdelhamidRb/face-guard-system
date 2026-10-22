"""
Package core pour le système de reconnaissance faciale FaceGuard.
"""

from .extractor import VGG16FeatureExtractor
from .detector import FaceDetector

__all__ = ["VGG16FeatureExtractor", "FaceDetector"]
