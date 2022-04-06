from collections import OrderedDict
from typing import List, Optional

import numpy as np
from algo_ops.ops.op import Op

from card_recognizer.classifier.core.card_prediction_result import CardPredictionResult
from card_recognizer.pulls_estimator.plots import plot_pull_stats
from card_recognizer.pulls_estimator.pull_stats import PullStats
from card_recognizer.reference.core.build import ReferenceBuild


class PullsEstimator(Op):
    """
    The PullsEstimator identifies the likely card pulls in a time series of image frames. It takes as input a time
    series of predictions of cards for frames in a video. The time series is represented as a CardPredictionResult
    object. The PullsEstimator filters out likely false positives in the time series based on frequencies of card
    detection and their confidence scores. The PullsEstimator then chooses the top-scoring cards based on the
    selection score: selection_score = card_frequency * confidence_score The PullsEstimator returns the estimated
    pulled cards in the video.
    """

    def __init__(
        self,
        freq_t: int = 5,
        conf_t: float = 0.1,
        num_cards_to_select: Optional[int] = 10,
        output_fig_path: Optional[str] = None,
        suppress_plotly_output: bool = True,
    ):
        super().__init__(func=self.estimate_pull_series)
        self.freq_t = freq_t
        self.conf_t = conf_t
        self.num_cards_to_select = num_cards_to_select
        self.output_fig_path = output_fig_path
        self.suppress_plotly_output = suppress_plotly_output

    def vis_input(self) -> None:
        """
        Visualize input statistics.
        """
        if self.input is not None:
            input_stats = self.tabulate_pull_statistics(
                frame_card_predictions=self.input
            )
            plot_pull_stats(
                pull_stats=input_stats,
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
            output_stats = self.tabulate_pull_statistics(
                frame_card_predictions=self.output
            )
            plot_pull_stats(
                pull_stats=output_stats,
                output_fig_path=self.output_fig_path,
                suppress_plotly_output=self.suppress_plotly_output,
                prefix="output",
            )

    def save_input(self, out_path: str) -> None:
        pass

    def save_output(self, out_path) -> None:
        pass

    @staticmethod
    def tabulate_unique_cards(
        frame_card_predictions: CardPredictionResult,
    ) -> List[int]:
        """
        Tabulate unique pulls (in temporal order using ordered dictionary)
        """
        unique_cards: List[int] = list(
            OrderedDict.fromkeys(
                [
                    frame_prediction.card_index_in_reference
                    for frame_prediction in frame_card_predictions
                ]
            ).keys()
        )
        return unique_cards

    @staticmethod
    def tabulate_pull_statistics(
        frame_card_predictions: CardPredictionResult,
    ) -> PullStats:
        """
        Tabulate pull statistics from pull series, returning a PullStats object.

        param frame_card_predictions: The time series of predicted cards in image frames

        Return:
            pull_stats: The statistics of pulls in the series
        """

        # obtain reference
        if frame_card_predictions.reference_set is not None:
            reference = ReferenceBuild.load(frame_card_predictions.reference_set)
        else:
            raise ValueError("Unspecified reference.")

        # tabulate unique cards in time series
        unique_cards = PullsEstimator.tabulate_unique_cards(
            frame_card_predictions=frame_card_predictions
        )

        # tabulate frequencies per unique pull
        card_frequencies: List[int] = [
            sum(
                [
                    1
                    for pull in frame_card_predictions
                    if pull.card_index_in_reference == find
                ]
            )
            for find in unique_cards
        ]

        # tabulate max confidence scores and selection scores per unique pull
        confidence_scores: List[List[float]] = list()
        max_confidence_scores: List[float] = list()
        selection_scores: List[float] = list()
        for i in range(len(unique_cards)):
            frames = [
                j
                for j, pull in enumerate(frame_card_predictions)
                if pull.card_index_in_reference == unique_cards[i]
            ]
            conf = [frame_card_predictions[f].conf for f in frames]
            max_conf_score = np.max(conf)
            confidence_scores.append(conf)
            max_confidence_scores.append(max_conf_score)
            selection_scores.append(float(max_conf_score * card_frequencies[i]))

        # return
        return PullStats(
            unique_cards=unique_cards,
            card_frequencies=card_frequencies,
            confidence_scores=confidence_scores,
            max_confidence_scores=max_confidence_scores,
            selection_scores=selection_scores,
            frame_card_predictions=frame_card_predictions,
            reference=reference,
        )

    def estimate_pull_series(
        self,
        frame_card_predictions: CardPredictionResult,
    ) -> CardPredictionResult:
        """
        Filters input frame card predictions using filter. Returns filtered card series.

        param input_card_frame_predictions: CardPredictionResult object containing card frame predictions

        Returns:
            Filtered CardPredictionResult with frames removed for filtered-out (false positive) cards
        """

        # tabulate pull stats for input
        pull_stats = self.tabulate_pull_statistics(
            frame_card_predictions=frame_card_predictions
        )

        # unpack tuple
        unique_cards = pull_stats.unique_cards
        card_frequencies = pull_stats.card_frequencies
        max_confidence_scores = pull_stats.max_confidence_scores
        selection_scores = pull_stats.selection_scores
        assert len(unique_cards) == len(card_frequencies)
        assert len(unique_cards) == len(max_confidence_scores)

        # run filter
        kept_cards_indices: List[int] = list()
        kept_selection_scores: List[float] = list()
        for i, card_index in enumerate(unique_cards):
            if (
                card_frequencies[i] >= self.freq_t
                and max_confidence_scores[i] >= self.conf_t
            ):
                kept_cards_indices.append(card_index)
                kept_selection_scores.append(selection_scores[i])

        # run selection rule
        sorted_card_indices = np.argsort(
            [-1.0 * kept_selection_scores[i] for i in range(len(kept_selection_scores))]
        )
        selected_cards = set([
            kept_cards_indices[index]
            for index in sorted_card_indices[0 : self.num_cards_to_select]
        ])

        # compute kept frames
        kept_frames = [
            pull
            for pull in frame_card_predictions
            if pull.card_index_in_reference in selected_cards
        ]

        # create output CardPredictionResult object
        rtn = CardPredictionResult(kept_frames)
        rtn.input_path = frame_card_predictions.input_path
        rtn.num_frames = frame_card_predictions.num_frames
        rtn.reference_set = frame_card_predictions.reference_set
        return rtn
