import os
import unittest

from card_recognizer.api.card_recognizer_pipeline import CardRecognizerPipeline
from card_recognizer.classifier.core.card_classification_result import (
    CardPredictionResult,
    CardPrediction,
)


class TestEndtoEnd(unittest.TestCase):
    def setUp(self) -> None:
        self.recognizer = CardRecognizerPipeline(set_name="Master")
        self.single_frames_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "single_images"
        )

    def test_end_to_end_klara(self) -> None:
        """
        Tests pipeline end to end on Klara image.
        """
        klara_path = os.path.join(self.single_frames_path, "klara.png")
        pred_result = self.recognizer.exec(inp=klara_path)
        self.assertTrue(isinstance(pred_result, CardPredictionResult))
        self.assertEqual(len(pred_result), 1)
        card_pred = pred_result[0]
        self.assertTrue(isinstance(card_pred, CardPrediction))
        card = self.recognizer.classifier.reference.lookup_card_prediction(
            card_prediction=card_pred
        )
        self.assertEqual(card.name, "Klara")
