import os
import unittest

import ezplotly.settings as plot_settings
from algo_ops.dependency.tester_util import clean_paths

from card_recognizer.api.card_recognizer import CardRecognizer, Mode
from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
)


class TestEndtoEnd(unittest.TestCase):
    def _clean_env(self) -> None:
        clean_paths(dirs=("out_figs",), files=("test.pkl",))

    def setUp(self) -> None:

        # suppress plotting for testing
        plot_settings.SUPPRESS_PLOTS = True

        # paths
        self.single_frames_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "single_images"
        )
        self._clean_env()

    def tearDown(self) -> None:
        self._clean_env()

    def test_end_to_end_klara(self) -> None:
        """
        Tests card recognizer end to end on Klara image.
        """

        # init card recognizer
        recognizer = CardRecognizer(set_name="master", mode=Mode.SINGLE_IMAGE)
        self.assertEqual(recognizer.input, None)
        self.assertEqual(recognizer.output, None)
        self.assertEqual(len(recognizer.execution_times), 0)

        # test on klara image
        klara_path = os.path.join(self.single_frames_path, "klara.png")
        pred_result = recognizer.exec(inp=klara_path)
        self.assertEqual(recognizer.input, klara_path)
        self.assertEqual(pred_result, recognizer.output)

        # check that there is only one result (Klara, frame 0),
        # and that the no-call frame was not returned as a result.
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

        # test pickle
        recognizer.to_pickle("test.pkl")

    def test_end_to_end_image_dir(self) -> None:
        """
        Tests card recognizer on images directory.
        """
        recognizer = CardRecognizer(set_name="master", mode=Mode.IMAGE_DIR)
        pred_result = recognizer.exec(inp=self.single_frames_path)

        # without filters, there should be two results. The first is Klara.
        self.assertTrue(isinstance(pred_result, CardPredictionResult))
        self.assertEqual(len(pred_result), 2)
        card_pred = pred_result[0]
        self.assertTrue(isinstance(card_pred, CardPrediction))
        self.assertEqual(card_pred.frame_index, 0)
        self.assertEqual(pred_result.num_frames, 2)
        card = recognizer.classifier.reference.lookup_card_prediction(
            card_prediction=card_pred
        )
        self.assertEqual(card.name, "Klara")

        # test pickle
        recognizer.to_pickle("test.pkl")

    def test_end_to_end_booster_dir(self) -> None:
        """
        Tests card recognizer on booster images directory.
        """
        recognizer = CardRecognizer(
            set_name="master",
            mode=Mode.BOOSTER_PULLS_IMAGE_DIR,
        )
        recognizer.set_output_path(output_path="out_figs")
        pred_result = recognizer.exec(inp=self.single_frames_path)
        self.assertEqual(len(pred_result), 2)
        self.assertEqual(pred_result[0], "Klara (#145) [0-1]")

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

        # test pickle
        recognizer.to_pickle("test.pkl")
