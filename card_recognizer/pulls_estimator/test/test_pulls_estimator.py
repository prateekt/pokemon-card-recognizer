import os
import shutil
import unittest

import pandas as pd
from algo_ops.pipeline.pipeline import Pipeline

from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
)
from card_recognizer.pulls_estimator.pulls_estimator import PullsEstimator
from card_recognizer.pulls_estimator.pulls_summary import PullsSummary


class TestPullsEstimator(unittest.TestCase):
    def setUp(self) -> None:

        # setup paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.pulls_filter = PullsEstimator(freq_t=5, conf_t=0.1, output_fig_path="figs")
        self.summary_file_path = os.path.join(dir_path, "summary_test.tsv")
        self.summary_file_path = os.path.join(dir_path, "summary_test.tsv")
        self.test_input_video_path = os.path.join(dir_path, "inp.avi")
        self.pulls_summary = PullsSummary(
            input_video=self.test_input_video_path, summary_file=self.summary_file_path
        )
        self.figs_path = os.path.join(dir_path, "figs")

        # make synthetic card prediction series
        predictions = [
            CardPrediction(card_index_in_reference=1, conf=0.9, frame_index=0),
            CardPrediction(card_index_in_reference=1, conf=0.8, frame_index=1),
            CardPrediction(card_index_in_reference=4, conf=0.05, frame_index=2),
            CardPrediction(card_index_in_reference=1, conf=0.8, frame_index=3),
            CardPrediction(card_index_in_reference=3, conf=0.9, frame_index=4),
            CardPrediction(card_index_in_reference=1, conf=0.9, frame_index=5),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=6),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=15),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=17),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=18),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=19),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=21),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=30),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=31),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=32),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=33),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=34),
            CardPrediction(card_index_in_reference=2, conf=0.9, frame_index=35),
        ]
        self.pred_series = CardPredictionResult(predictions=predictions, num_frames=50)
        self.pred_series.reference_set = "Brilliant Stars"

    def test_pulls_filter(self) -> None:
        """
        Test basic functionality.
        """

        # run filter
        self.assertEqual(len(self.pred_series.runs), 6)
        output_preds = self.pulls_filter.estimate_pull_series(
            frame_card_predictions=self.pred_series
        )
        self.assertEqual(len(output_preds.runs), 3)
        pulls_summary = self.pulls_summary.make_pulls_summary(
            card_predictions=output_preds
        )
        self.assertListEqual(
            pulls_summary,
            [
                "Exeggutor (#2) [0-6]",
                "Shroomish (#3) [15-22]",
                "Shroomish (#3) [30-36]",
            ],
        )

        # check generated summary file
        self.assertTrue(os.path.exists(self.summary_file_path))
        df = pd.read_csv(self.summary_file_path, sep="\t")
        self.assertEqual(len(df), 1)
        self.assertEqual(len(df.columns), 4)
        os.unlink(self.summary_file_path)

    def test_pulls_estimation_pipeline(self) -> None:
        """
        Test as Pipeline.
        """

        # setup pulls estimation pipeline
        pulls_pipeline = Pipeline(ops=[self.pulls_filter, self.pulls_summary])

        # run
        result = pulls_pipeline.exec(inp=self.pred_series)
        self.assertListEqual(
            result,
            [
                "Exeggutor (#2) [0-6]",
                "Shroomish (#3) [15-22]",
                "Shroomish (#3) [30-36]",
            ],
        )

        # visualize
        pulls_pipeline.vis()
        self.assertTrue(os.path.exists(os.path.join("figs", "input_metrics.png")))
        self.assertTrue(os.path.exists(os.path.join("figs", "output_metrics.png")))
        self.assertTrue(
            os.path.exists(
                os.path.join("figs", "input_frame_prediction_time_series.png")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join("figs", "output_frame_prediction_time_series.png")
            )
        )

        # check generated summary file
        self.assertTrue(os.path.exists(self.summary_file_path))
        df = pd.read_csv(self.summary_file_path, sep="\t")
        self.assertEqual(len(df), 1)
        self.assertEqual(len(df.columns), 4)

    def tearDown(self) -> None:
        if os.path.exists(self.summary_file_path):
            os.unlink(self.summary_file_path)
        if os.path.exists(self.figs_path):
            shutil.rmtree(self.figs_path)
