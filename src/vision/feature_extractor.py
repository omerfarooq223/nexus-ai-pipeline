"""Visual feature extraction using OpenCV and MediaPipe."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class VisualFeatures:
    """Features extracted from one image."""

    face_count: int
    edge_density: float
    brightness_mean: float


class VisualFeatureExtractor:
    """Combines OpenCV image stats with MediaPipe face detection."""

    def __init__(self) -> None:
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

    def extract(self, image_bgr: np.ndarray) -> VisualFeatures:
        """Extract robust and lightweight visual features from BGR image."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detection = self.face_detector.process(image_rgb)
        face_count = 0
        if detection.detections:
            face_count = len(detection.detections)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        brightness_mean = float(np.mean(gray))

        return VisualFeatures(
            face_count=face_count,
            edge_density=edge_density,
            brightness_mean=brightness_mean,
        )

    @staticmethod
    def to_dict(features: VisualFeatures) -> Dict[str, Any]:
        """Convert dataclass to serializable dictionary."""
        return {
            "face_count": features.face_count,
            "edge_density": features.edge_density,
            "brightness_mean": features.brightness_mean,
        }
