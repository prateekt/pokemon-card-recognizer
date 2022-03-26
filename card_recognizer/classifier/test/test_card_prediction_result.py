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
