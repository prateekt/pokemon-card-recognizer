import os
import unittest

from card_recognizer.api.card_recognizer_pipeline import CardRecognizerPipeline, Mode
from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
)


class TestEndtoEnd(unittest.TestCase):
    def setUp(self) -> None:
        self.single_frames_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "single_images"
        )

    def test_end_to_end_klara(self) -> None:
        """
        Tests card recognizer end to end on Klara image.
        """
        recognizer = CardRecognizerPipeline(set_name="master", mode=Mode.SINGLE_IMAGE)
        klara_path = os.path.join(self.single_frames_path, "klara.png")
        pred_result = recognizer.exec(inp=klara_path)
        self.assertTrue(isinstance(pred_result, CardPredictionResult))
        self.assertEqual(len(pred_result), 1)
        card_pred = pred_result[0]
        self.assertTrue(isinstance(card_pred, CardPrediction))
        self.assertEqual(card_pred.frame_index, None)
        card = recognizer.classifier.reference.lookup_card_prediction(
            card_prediction=card_pred
        )
        self.assertEqual(card.name, "Klara")

    def test_end_to_end_image_dir(self) -> None:
        """
        Tests card recognizer on images directory.
        """
        recognizer = CardRecognizerPipeline(set_name="master", mode=Mode.IMAGE_DIR)
        pred_result = recognizer.exec(inp=self.single_frames_path)

        # check that there is only one result (Klara, frame 0),
        # and that the no-call frame was not returned as a result.
        self.assertTrue(isinstance(pred_result, CardPredictionResult))
        self.assertEqual(len(pred_result), 1)
        card_pred = pred_result[0]
        self.assertTrue(isinstance(card_pred, CardPrediction))
        self.assertEqual(card_pred.frame_index, 0)
        card = recognizer.classifier.reference.lookup_card_prediction(
            card_prediction=card_pred
        )
        self.assertEqual(card.name, "Klara")
