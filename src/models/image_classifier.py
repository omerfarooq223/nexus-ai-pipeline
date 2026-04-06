"""Image classification with TensorFlow/Keras pre-trained CNN."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)


logger = logging.getLogger(__name__)


@dataclass
class ImagePrediction:
    """Top-1 prediction result from the CNN model."""

    label: str
    confidence: float


class ImageClassifier:
    """MobileNetV2 classifier wrapper."""

    def __init__(self) -> None:
        self.model: Optional[MobileNetV2] = None
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Load model lazily so startup does not fail on artifact/network issues."""
        if self._model_loaded:
            return

        try:
            logger.info("Loading MobileNetV2 model weights=imagenet")
            self.model = MobileNetV2(weights="imagenet")
            self._model_loaded = True
            logger.info("MobileNetV2 model loaded successfully")
        except Exception as exc:
            logger.error("Model load failed: %s", exc, exc_info=True)
            raise

    def classify(self, image_bgr: np.ndarray) -> ImagePrediction:
        """Resize and classify input image."""
        self._ensure_model_loaded()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (224, 224))
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        batch = preprocess_input(batch)

        if self.model is None:
            raise RuntimeError("Image model is not loaded")

        probs = self.model.predict(batch, verbose=0)
        top = decode_predictions(probs, top=1)[0][0]

        return ImagePrediction(label=top[1], confidence=float(top[2]))
