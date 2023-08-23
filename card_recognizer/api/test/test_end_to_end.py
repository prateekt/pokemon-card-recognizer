import os
import unittest

import ezplotly.settings as plot_settings
import pandas as pd
from algo_ops.dependency.tester_util import clean_paths
from ocr_ops.framework.op.ffmpeg_op import FFMPEGOp

from card_recognizer.api.card_recognizer import CardRecognizer, OperatingMode
from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
)


class TestEndToEnd(unittest.TestCase):
    @staticmethod
    def _clean_env() -> None:
        clean_paths(dirs=("out_figs",), files=("test.pkl", "summary.txt"))

    def setUp(self) -> None:
        # suppress plotting for testing
        plot_settings.SUPPRESS_PLOTS = True

        # paths
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.single_frames_path = os.path.join(dir_path, "single_images")
        self.video_path = os.path.join(dir_path, "video", "test.avi")
        self._clean_env()

    def tearDown(self) -> None:
        self._clean_env()

    def test_end_to_end_single_image(self) -> None:
        """
        Tests card recognizer end-to-end on single image.
        """

        # init card recognizer
        recognizer = CardRecognizer(set_name="master", mode=OperatingMode.SINGLE_IMAGE)
        self.assertEqual(recognizer.input, None)
        self.assertEqual(recognizer.output, None)
        self.assertEqual(len(recognizer.execution_times), 0)

        # test on image
        image_path = os.path.join(self.single_frames_path, "card.png")
        pred_result = recognizer.exec(inp=image_path)
        self.assertTrue(isinstance(recognizer.input, str))
        self.assertTrue(isinstance(recognizer.output, CardPredictionResult))
        self.assertEqual(recognizer.input, image_path)
        self.assertEqual(pred_result, recognizer.output)

        # check that there is only one result (frame 0),
        # and that the no-call frame was not returned as a result.
        self.assertTrue(isinstance(pred_result, CardPredictionResult))
        self.assertEqual(len(pred_result), 1)
        self.assertTrue(isinstance(pred_result[0], CardPrediction))
        detected_card = recognizer.classifier.reference.lookup_card_prediction(
            card_prediction=pred_result[0]
        )
        self.assertEqual(detected_card.name, "Sprigatito")
        self.assertEqual(detected_card.set.name, "Paldea Evolved")
        self.assertEqual(int(detected_card.number), 12)

        # test pickle
        recognizer.to_pickle("test.pkl")

    def test_end_to_end_image_dir(self) -> None:
        """
        Tests card recognizer on directory of images.
        """
        recognizer = CardRecognizer(set_name="master", mode=OperatingMode.IMAGE_DIR)
        pred_result = recognizer.exec(inp=self.single_frames_path)

        # without filters, there should be two results. The first is the card.
        self.assertTrue(isinstance(pred_result, CardPredictionResult))
        self.assertEqual(len(pred_result), 2)
        card_pred = pred_result[0]
        self.assertTrue(isinstance(card_pred, CardPrediction))
        self.assertEqual(card_pred.frame_index, 0)
        self.assertEqual(pred_result.num_frames, 2)
        card = recognizer.classifier.reference.lookup_card_prediction(
            card_prediction=card_pred
        )
        self.assertEqual(card.name, "Sprigatito")

        # test pickle
        recognizer.to_pickle("test.pkl")

    def test_end_to_end_booster_dir(self) -> None:
        """
        Tests card recognizer on booster images directory.
        """
        recognizer = CardRecognizer(
            set_name="master",
            mode=OperatingMode.BOOSTER_PULLS_IMAGE_DIR,
            min_run_length=None,
            min_run_conf=0.05,
        )
        recognizer.set_output_path(output_path="out_figs")
        pred_result = recognizer.exec(inp=self.single_frames_path)

        # With filters, only one result.
        self.assertEqual(len(pred_result), 1)
        self.assertEqual(pred_result[0], "Sprigatito (Paldea Evolved #12) [0-1]")

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

    def test_end_to_end_video(self) -> None:
        """
        Test card recognizer on video.
        """

        # setup recognizer
        recognizer = CardRecognizer(
            set_name="Chilling Reign",
            mode=OperatingMode.PULLS_VIDEO,
            min_run_length=None,
            min_run_conf=0.05,
        )
        ffmpeg_op = recognizer.ops[list(recognizer.ops.keys())[0]]
        assert isinstance(ffmpeg_op, FFMPEGOp)
        ffmpeg_op.fps = 1
        recognizer.set_output_path(output_path="out_figs")
        recognizer.set_summary_file(summary_file="summary.txt")

        # test running
        pred_result = recognizer.exec(inp=self.video_path)
        self.assertEqual(recognizer.input, self.video_path)
        self.assertEqual(recognizer.output, pred_result)
        self.assertEqual(pred_result, ["Klara (#145) [1-2]"])
        self.assertEqual(len(pred_result), 1)

        # check created directory structure
        self.assertTrue(os.path.exists("out_figs"))
        self.assertTrue(
            os.path.exists(os.path.join("out_figs", "uncompressed_video_frames"))
        )
        self.assertEqual(
            len(os.listdir(os.path.join("out_figs", "uncompressed_video_frames"))), 3
        )

        # check summary file
        self.assertTrue(os.path.exists("summary.txt"))
        summary_df = pd.read_csv("summary.txt", sep="\t")
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.columns.to_list(), ["input_path", "P_1"])
        self.assertEqual(summary_df.input_path[0], self.video_path)
        self.assertEqual(summary_df.P_1[0], "Klara (#145) [1-2]")
