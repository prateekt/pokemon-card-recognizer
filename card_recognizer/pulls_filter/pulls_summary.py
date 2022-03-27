from typing import List

from algo_ops.ops.text import TextOp

from card_recognizer.classifier.core.card_prediction_result import CardPredictionResult
from card_recognizer.pulls_filter.pulls_filter import PullsFilter
from card_recognizer.reference.core.build import ReferenceBuild


class PullsSummary(TextOp):
    """
    Converts a time series of frame card predictions into a simple summary of pulled cards.
    """

    @staticmethod
    def pulls_summary(
        frame_card_predictions: CardPredictionResult,
    ) -> List[str]:
        """
        Simply lists the unique pulled cards.

        param frame_card_predictions: The card predictions in frames.

        Returns:
            Listing of pulled Pokemon cards
        """

        # obtain reference
        if frame_card_predictions.reference_set is not None:
            reference = ReferenceBuild.load(frame_card_predictions.reference_set)
        else:
            raise ValueError("Unspecified reference.")

        # obtain unique cards
        unique_cards = PullsFilter.tabulate_unique_cards(
            frame_card_predictions=frame_card_predictions
        )
        unique_card_names = [
            reference.cards[pull].name + " (#" + str(reference.cards[pull].number) + ")"
            for pull in unique_cards
        ]
        return unique_card_names

    def __init__(self):
        super().__init__(func=self.pulls_summary)
