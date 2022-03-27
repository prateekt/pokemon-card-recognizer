import os
import shutil
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
        self.assertEqual(pred_result.num_frames, None)
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
        self.assertEqual(pred_result.num_frames, 2)
        card = recognizer.classifier.reference.lookup_card_prediction(
            card_prediction=card_pred
        )
        self.assertEqual(card.name, "Klara")

    def test_end_to_end_booster_dir(self) -> None:
        """
        Tests card recognizer on booster images directory.
        """
        recognizer = CardRecognizerPipeline(
            set_name="master",
            mode=Mode.BOOSTER_PULLS_IMAGE_DIR,
        )
        recognizer.set_output_figs_path(output_figs_path='out_figs')
        pred_result = recognizer.exec(inp=self.single_frames_path)
        self.assertEqual(len(pred_result), 1)
        self.assertEqual(pred_result, ["Klara (#145)"])

        # test visualization capability and plot generation
        recognizer.vis()
        self.assertTrue(os.path.exists("out_figs"))
        self.assertTrue(
            os.path.exists(
                os.path.join("out_figs", "input_frame_prediction_time_series.png")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join("out_figs", "output_frame_prediction_time_series.png")
            )
        )
        self.assertTrue(os.path.exists(os.path.join("out_figs", "input_metrics.png")))
        self.assertTrue(os.path.exists(os.path.join("out_figs", "output_metrics.png")))
        shutil.rmtree("out_figs")
