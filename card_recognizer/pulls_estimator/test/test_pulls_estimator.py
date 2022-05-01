import os
import shutil
import unittest

import pandas as pd
from algo_ops.pipeline.pipeline import Pipeline

from card_recognizer.classifier.core.card_prediction_result import CardPredictionResult
from card_recognizer.pulls_estimator.pulls_estimator import PullsEstimator
from card_recognizer.pulls_estimator.pulls_summary import PullsSummary


class TestPullsEstimator(unittest.TestCase):
    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.sample_dir = os.path.join(dir_path, "sample_data")
        self.pulls_filter = PullsEstimator(freq_t=5, conf_t=0.1, output_fig_path="figs")
        self.summary_file_path = 'summary_test.tsv'
        self.pulls_summary = PullsSummary(self.summary_file_path)
        if os.path.exists(self.summary_file_path):
            os.unlink(self.summary_file_path)

    def test_pulls_filter(self) -> None:
        """
        Test basic functionality.
        """

        # load example card prediction series
        sample_file = os.path.join(self.sample_dir, "VID_20220316_214213.mp4.pkl")
        frame_card_predictions = CardPredictionResult.load_from_pickle(
            pkl_path=sample_file
        )
        frame_card_predictions.reference_set = "Master"

        # run filter
        output_preds = self.pulls_filter.estimate_pull_series(
            frame_card_predictions=frame_card_predictions
        )
        pulls_summary = self.pulls_summary.pulls_summary(
            frame_card_predictions=output_preds
        )
        self.assertEqual(len(pulls_summary), 10)

        # check file
        self.assertTrue(os.path.exists(self.summary_file_path))
        df = pd.read_csv(self.summary_file_path, sep='\t')
        self.assertEqual(len(df), 1)
        self.assertEqual(len(df.columns), 10)
        os.unlink(self.summary_file_path)

    def test_pulls_estimation_pipeline(self) -> None:
        """
        Test as Pipeline.
        """

        # load example card prediction series
        sample_file = os.path.join(self.sample_dir, "VID_20220316_214213.mp4.pkl")
        frame_card_predictions = CardPredictionResult.load_from_pickle(
            pkl_path=sample_file
        )
        frame_card_predictions.reference_set = "Master"

        # setup pulls estimation pipeline
        pulls_pipeline = Pipeline(ops=[self.pulls_filter, self.pulls_summary])

        # run
        result = pulls_pipeline.exec(inp=frame_card_predictions)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 10)

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
        shutil.rmtree("figs")
