from typing import List, Optional

import numpy as np
from algo_ops.ops.op import Op

from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
    Run,
)
from card_recognizer.pulls_estimator.plots import plot_pull_stats


class PullsEstimator(Op):
    """
    The PullsEstimator identifies the likely card pulls in a time series of image frames. It takes as input a time
    series of predictions of cards for frames in a video. The time series is represented as a CardPredictionResult
    object. The PullsEstimator filters out likely false positives in the time series based on frequencies of card
    detection and their confidence scores. The PullsEstimator then chooses the top-scoring cards based on the
    selection score: selection_score = card_frequency * confidence_score. The PullsEstimator returns the estimated
    pulled cards in the video.
    """

    def __init__(
        self,
        freq_t: Optional[int] = 5,
        conf_t: Optional[float] = 0.1,
        num_cards_to_select: Optional[int] = 10,
        output_fig_path: Optional[str] = None,
        suppress_plotly_output: bool = True,
        run_tol: Optional[int] = 10,
    ):
        super().__init__(func=self.estimate_pull_series)
        self.freq_t: Optional[int] = freq_t
        self.conf_t: Optional[float] = conf_t
        self.num_cards_to_select = num_cards_to_select
        self.output_fig_path = output_fig_path
        self.suppress_plotly_output = suppress_plotly_output
        self.run_tol = run_tol

    def vis_input(self) -> None:
        """
        Visualize input statistics.
        """
        if self.input is not None:
            plot_pull_stats(
                card_prediction_result=self.input,
                output_fig_path=self.output_fig_path,
                suppress_plotly_output=self.suppress_plotly_output,
                prefix="input",
            )

    def vis(self) -> None:
        """
        Visualize output statistics.
        """
        self.vis_input()
        if self.output is not None:
            plot_pull_stats(
                card_prediction_result=self.output,
                output_fig_path=self.output_fig_path,
                suppress_plotly_output=self.suppress_plotly_output,
                prefix="output",
            )

    def save_input(self, out_path: str) -> None:
        pass

    def save_output(self, out_path) -> None:
        pass

    def _apply_filter(self, runs: List[Run]) -> List[Run]:
        """
        Filters a list of card runs on run length and max confidence score.

        param runs: The input list of card runs

        Returns:
            keep: List of kept runs
        """
        keep: List[Run] = list()
        for run in runs:
            if (self.freq_t is not None and len(run) < self.freq_t) or (
                self.conf_t is not None and run.max_confidence_score < self.conf_t
            ):
                continue
            else:
                keep.append(run)
        return keep

    def _apply_selection(self, runs: List[Run]) -> List[Run]:
        """
        Chooses a number of top runs based on selection score.

        param runs: List of candidate runs

        Returns:
            selected_runs: List of selected runs
        """

        # if num_cards_to_select is None, return all runs
        if self.num_cards_to_select is None:
            return runs

        # perform selection based on selection scores
        sorted_card_indices = np.argsort([-1.0 * run.selection_score for run in runs])
        selected_runs: List[Run] = [
            runs[index] for index in sorted_card_indices[0 : self.num_cards_to_select]
        ]
        return selected_runs

    @staticmethod
    def _make_result(
        runs: List[Run], frame_card_prediction: CardPredictionResult
    ) -> CardPredictionResult:
        """
        Packages runs to CardPredictionResult.

        params runs: List of runs
        param frame_card_prediction: Previous CardPredictionResult object

        Returns:
            New card prediction result with just specified runs kept
        """

        # find kept predictions from selected runs
        kept_predictions: List[CardPrediction] = list()
        for run in runs:
            kept_predictions.extend(
                frame_card_prediction.query_card_prediction(
                    interval=run.interval, card_index=run.card_index
                )
            )
        kept_predictions.sort()

        # create return object
        rtn = CardPredictionResult(
            predictions=kept_predictions, num_frames=frame_card_prediction.num_frames
        )
        rtn.input_path = frame_card_prediction.input_path
        rtn.reference_set = frame_card_prediction.reference_set
        return rtn

    def estimate_pull_series(
        self,
        frame_card_predictions: CardPredictionResult,
    ) -> CardPredictionResult:
        """
        Estimates

        param input_card_frame_predictions: CardPredictionResult object containing card frame predictions

        Returns:
            Filtered CardPredictionResult with frames removed for filtered-out (false positive) cards
        """

        # apply filter
        kept_runs: List[Run] = self._apply_filter(runs=frame_card_predictions.runs)

        # select cards based on selection score
        selected_runs: List[Run] = self._apply_selection(runs=kept_runs)

        # create output CardPredictionResult object
        return self._make_result(
            runs=selected_runs, frame_card_prediction=frame_card_predictions
        )
