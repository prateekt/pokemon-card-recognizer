import os
import unittest

from card_recognizer.classifier.core.card_prediction_result import (
    CardPrediction,
    CardPredictionResult,
)


class TestCardPredictionResult(unittest.TestCase):
    def test_list_usage(self) -> None:
        """
        Test using CardPredictionResult in list mode.
        """
        pred1 = CardPrediction(card_index_in_reference=0, conf=0)
        pred2 = CardPrediction(card_index_in_reference=1, conf=0.5)
        pred3 = CardPrediction(card_index_in_reference=3, conf=1.0)
        result = CardPredictionResult(predictions=[pred1, pred2, pred3])
        self.assertTrue(isinstance(result, CardPredictionResult))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], pred1)
        self.assertEqual(result[1], pred2)
        self.assertEqual(result[2], pred3)
        for i, r in enumerate(result):
            if i == 0:
                self.assertEqual(r, pred1)
            elif i == 1:
                self.assertEqual(r, pred2)
            elif i == 2:
                self.assertEqual(r, pred3)

    def test_pickle(self) -> None:
        """
        Test pickle capability.
        """
        pred1 = CardPrediction(card_index_in_reference=0, conf=0)
        pred2 = CardPrediction(card_index_in_reference=1, conf=0.5)
        pred3 = CardPrediction(card_index_in_reference=3, conf=1.0)
        result = CardPredictionResult(predictions=[pred1, pred2, pred3])
        result.to_pickle("test.pkl")
        self.assertTrue(os.path.exists("test.pkl"))
        loaded_obj = CardPredictionResult.load_from_pickle(pkl_path="test.pkl")
        self.assertTrue(isinstance(loaded_obj, CardPredictionResult))
        self.assertTrue(loaded_obj is not result)
        self.assertEqual(len(loaded_obj), len(result))
        self.assertTrue(len(loaded_obj), 3)
        self.assertEqual(loaded_obj[1].frame_index, None)
        self.assertEqual(loaded_obj[2].conf, 1.0)
        os.unlink("test.pkl")
