import os
import random
import unittest

import ezplotly.settings as plot_settings
from algo_ops.dependency.tester_util import clean_paths

from card_recognizer.classifier.core.card_prediction_result import (
    CardPrediction,
    CardPredictionResult,
)
from card_recognizer.pulls_estimator.plots import plot_paged_metrics


class TestPlotPaging(unittest.TestCase):
    def setUp(self) -> None:
        """
        Setup synthetic time series to test plot paging.
        """

        # setup plot settings for testing
        plot_settings.SUPPRESS_PLOTS = True

        # setup paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.figs_path = os.path.join(dir_path, "figs")
        os.makedirs(self.figs_path, exist_ok=True)

        # make synthetic card prediction series
        num_frames = 33
        predictions = [
            CardPrediction(
                card_index_in_reference=random.randint(0, 3), conf=0.9, frame_index=i
            )
            for i in range(num_frames)
        ]
        self.pred_series = CardPredictionResult(
            predictions=predictions, num_frames=num_frames, run_tol=0
        )
        self.pred_series.reference_set = "Master"
        self.clear_plots = True

    def test_paging(self):
        """
        Plot paged metrics and test correct number of and names of files exist.
        """
        plot_paged_metrics(
            frame_card_predictions=self.pred_series,
            outfile=os.path.join(self.figs_path, "metrics.png"),
        )
        for i in [1, 2, 3]:
            self.assertTrue(
                os.path.exists(
                    os.path.join(self.figs_path, "metrics_page" + str(i) + ".png")
                )
            )

    def tearDown(self) -> None:
        if self.clear_plots:
            clean_paths(dirs=(self.figs_path,))
