"""End-to-end multimodal orchestration service."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any

import cv2
import numpy as np

from src.db.mysql_client import InferenceRecord, MySQLClient
from src.models.image_classifier import ImageClassifier
from src.nlp.text_processor import TextProcessor
from src.vision.feature_extractor import VisualFeatureExtractor


class MultiModalService:
    """Combines image and text inference into one response."""

    def __init__(self) -> None:
        self.vision_extractor = VisualFeatureExtractor()
        self.text_processor = TextProcessor()
        self.image_classifier = ImageClassifier()
        self.db = MySQLClient()

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """Decode image bytes into OpenCV BGR array."""
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image bytes")
        return image

    def run_inference(self, image_name: str, image_bytes: bytes, query_text: str) -> Dict[str, Any]:
        """Run all model components and persist inference output."""
        image = self._decode_image(image_bytes)

        visual_features = self.vision_extractor.extract(image)
        text_features = self.text_processor.process(query_text)
        image_prediction = self.image_classifier.classify(image)

        summary = (
            f"Image likely contains '{image_prediction.label}' "
            f"(confidence={image_prediction.confidence:.3f}). "
            f"Query sentiment is {text_features.sentiment_label} "
            f"(score={text_features.sentiment_score:.3f}) with "
            f"{visual_features.face_count} detected faces."
        )

        record = InferenceRecord(
            image_name=image_name,
            query_text=query_text,
            image_label=image_prediction.label,
            image_confidence=image_prediction.confidence,
            face_count=visual_features.face_count,
            edge_density=visual_features.edge_density,
            token_count=len(text_features.tokens),
            sentiment_label=text_features.sentiment_label,
            sentiment_score=text_features.sentiment_score,
            combined_summary=summary,
        )
        inserted_id = self.db.insert_inference(record)

        return {
            "id": inserted_id,
            "image_prediction": asdict(image_prediction),
            "visual_features": self.vision_extractor.to_dict(visual_features),
            "text_features": {
                "tokens": text_features.tokens,
                "lemmas": text_features.lemmas,
                "noun_phrases": text_features.noun_phrases,
                "sentiment_label": text_features.sentiment_label,
                "sentiment_score": text_features.sentiment_score,
            },
            "summary": summary,
        }
