import os
from typing import List, Optional

from algo_ops.ops.text import TextOp

from card_recognizer.classifier.core.card_prediction_result import CardPredictionResult
from card_recognizer.pulls_estimator.pulls_estimator import PullsEstimator
from card_recognizer.reference.core.build import ReferenceBuild


class PullsSummary(TextOp):
    """
    Converts a time series of frame card predictions into a simple summary of pulled cards.
    """

    def pulls_summary(
        self,
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
        unique_cards = PullsEstimator.tabulate_unique_cards(
            frame_card_predictions=frame_card_predictions
        )
        unique_card_names = [
            reference.cards[pull].name + " (#" + str(reference.cards[pull].number) + ")"
            for pull in unique_cards
        ]

        # write row to summary file (if specified)
        if self.summary_file is not None:
            if not os.path.exists(self.summary_file):
                first_write = True
            else:
                first_write = False
            with open(self.summary_file, 'a') as fout:
                if first_write:
                    header_cols = ['P_'+str(i+1) for i in range(len(unique_card_names))]
                    header = '\t'.join(header_cols)+'\n'
                    fout.write(header)
                line = '\t'.join(unique_card_names)+'\n'
                fout.write(line)

        # return
        return unique_card_names

    def __init__(self, summary_file: Optional[str] = None):
        super().__init__(func=self.pulls_summary)
        self.summary_file = summary_file
