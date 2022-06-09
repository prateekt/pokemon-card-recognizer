import os
import unittest

import ezplotly.settings as plot_settings
import pandas as pd
from algo_ops.dependency.tester_util import clean_paths
from algo_ops.pipeline.pipeline import Pipeline

from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
)
from card_recognizer.pulls_estimator.pulls_estimator import PullsEstimator
from card_recognizer.pulls_estimator.pulls_summary import PullsSummary


class TestPullsEstimator(unittest.TestCase):
    @staticmethod
    def _clean_env() -> None:
        clean_paths(
            dirs=("figs", "pulls_pipeline_profile", "test_output_save"),
            files=(
                "summary_test.tsv",
                "test.pkl",
            ),
        )

    def setUp(self) -> None:

        # disable dynamic plotting
        plot_settings.SUPPRESS_PLOTS = True

        # setup paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_input_video_path = os.path.join(dir_path, "inp.avi")
        self._clean_env()

        # setup pipeline components
        self.pulls_estimator = PullsEstimator(
            min_run_length=5, min_run_conf=0.1, output_figs_path="figs"
        )
        self.pulls_summary = PullsSummary(summary_file="summary_test.tsv")

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
        self.pred_series.input_path = "example.avi"

    def test_pulls_estimator(self) -> None:
        """
        Test basic functionality of pulls estimator.
        """

        # run filter
        self.assertEqual(len(self.pred_series.runs), 6)
        output_preds = self.pulls_estimator.estimate_pull_series(
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
        self.assertTrue(os.path.exists("summary_test.tsv"))
        df = pd.read_csv("summary_test.tsv", sep="\t")
        self.assertEqual(len(df), 1)
        self.assertEqual(len(df.columns), 4)
        self.assertListEqual(df.columns.to_list(), ["input_path", "P_1", "P_2", "P_3"])
        self.assertEqual(df.input_path[0], "example.avi")
        self.assertEqual(df.P_1[0], "Exeggutor (#2) [0-6]")
        self.assertEqual(df.P_2[0], "Shroomish (#3) [15-22]")
        self.assertEqual(df.P_3[0], "Shroomish (#3) [30-36]")

    def test_pulls_estimation_pipeline(self) -> None:
        """
        Test as Pipeline.
        """

        # setup pulls estimation pipeline
        pulls_pipeline = Pipeline(ops=[self.pulls_estimator, self.pulls_summary])
        self.assertEqual(pulls_pipeline.input, None)
        self.assertEqual(pulls_pipeline.output, None)
        self.assertTrue(pulls_pipeline.ops, [self.pulls_estimator, self.pulls_summary])
        for method in (
            pulls_pipeline.vis,
            pulls_pipeline.vis_input,
            pulls_pipeline.save_input,
            pulls_pipeline.save_output,
            pulls_pipeline.vis_profile,
        ):
            with self.assertRaises(ValueError):
                method()

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
        with self.assertRaises(ValueError):
            pulls_pipeline.vis_input()
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

        # save input/output
        with self.assertRaises(ValueError):
            pulls_pipeline.save_input()
        pulls_pipeline.save_output(out_path="test_output_save")
        self.assertEqual(len(os.listdir("test_output_save")), 3)

        # vis profile
        pulls_pipeline.vis_profile(profiling_figs_path="pulls_pipeline_profile")
        for file in (
            "['estimate_pull_series', 'make_pulls_summary']",
            "['estimate_pull_series', 'make_pulls_summary']_violin",
            "estimate_pull_series",
            "make_pulls_summary",
        ):
            self.assertTrue(
                os.path.exists(os.path.join("pulls_pipeline_profile", file + ".png"))
            )

        # check generated summary file
        self.assertTrue(os.path.exists("summary_test.tsv"))
        df = pd.read_csv("summary_test.tsv", sep="\t")
        self.assertEqual(len(df), 1)
        self.assertEqual(len(df.columns), 4)
        self.assertEqual(len(df), 1)
        self.assertEqual(len(df.columns), 4)
        self.assertListEqual(df.columns.to_list(), ["input_path", "P_1", "P_2", "P_3"])
        self.assertEqual(df.input_path[0], "example.avi")
        self.assertEqual(df.P_1[0], "Exeggutor (#2) [0-6]")
        self.assertEqual(df.P_2[0], "Shroomish (#3) [15-22]")
        self.assertEqual(df.P_3[0], "Shroomish (#3) [30-36]")

        # test pickle
        pulls_pipeline.to_pickle("test.pkl")
        self.assertTrue(os.path.exists("test.pkl"))

    def tearDown(self) -> None:
        self._clean_env()
