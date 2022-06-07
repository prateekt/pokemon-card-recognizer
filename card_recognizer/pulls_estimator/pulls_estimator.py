from typing import List, Optional

import numpy as np
from algo_ops.ops.text import TextOp

from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
    Run,
)
from card_recognizer.pulls_estimator.plots import plot_pull_stats


class PullsEstimator(TextOp):
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
        min_run_length: Optional[int] = 5,
        min_run_conf: Optional[float] = 0.1,
        run_tol: Optional[int] = 10,
        num_cards_to_select: Optional[int] = 10,
        output_figs_path: Optional[str] = None,
        figs_paging: bool = False,
    ):
        """
        param freq_t: The minimum length of a run to keep it as a card run detection (if None,
            turn off filter and allow all runs to pass)
        param conf_t: The minimum confidence score of a run to keep it as a card run detection (if None,
            turn off filter and allow all runs to pass)
        param run_tol: The number of consecutive noisy frames to tolerate within a run
        param num_cards_to_select; The number of card pulls to estimate (if None, there is no limit)
        param output_figs_path: Path to where output figs should go
        figs_paging: Whether figs should be paged
        """

        # set params
        super().__init__(func=self.estimate_pull_series)
        self.freq_t: Optional[int] = min_run_length
        self.conf_t: Optional[float] = min_run_conf
        self.run_tol: Optional[int] = run_tol
        self.num_cards_to_select = num_cards_to_select
        self.output_fig_path = output_figs_path
        self.figs_paging = figs_paging

        # define input/output types
        self.input: Optional[CardPredictionResult] = None
        self.output: Optional[CardPredictionResult] = None

    def vis_input(self) -> None:
        """
        Visualize input statistics.
        """
        if self.input is None:
            raise ValueError("There is no input to be visualized.")
        plot_pull_stats(
            card_prediction_result=self.input,
            output_fig_path=self.output_fig_path,
            prefix="input",
            figs_paging=self.figs_paging,
        )

    def vis(self) -> None:
        """
        Visualize output statistics.
        """
        if self.output is None:
            raise ValueError("There is no output to be visualized.")
        self.vis_input()
        plot_pull_stats(
            card_prediction_result=self.output,
            output_fig_path=self.output_fig_path,
            prefix="output",
            figs_paging=self.figs_paging,
        )

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
        Estimates series of pulled cards in a stream of card detections in images.

        param frame_card_predictions: CardPredictionResult object containing card frame predictions

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
