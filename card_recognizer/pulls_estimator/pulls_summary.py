import os
from typing import List, Optional

from algo_ops.ops.text import TextOp

from card_recognizer.classifier.core.card_prediction_result import CardPredictionResult
from card_recognizer.reference.core.build import ReferenceBuild


class PullsSummary(TextOp):
    """
    Converts a time series of frame card predictions into a simple summary of pulled cards.
    """

    def __init__(
        self,
        summary_file: Optional[str] = None,
    ):
        """
        param summary_file: Path to where summary file should be written. If None, no summary file is written.
        param input_file: If applicable, path to input file being processed (e.g. a video file).
        """

        # set params
        super().__init__(func=self.make_pulls_summary)
        self.summary_file = summary_file

        # define input/output types
        self.input: Optional[CardPredictionResult] = None
        self.output: Optional[List[str]] = None

    def make_pulls_summary(
        self,
        card_predictions: CardPredictionResult,
    ) -> List[str]:
        """
        Simply lists the unique pulled cards in the runs.

        param card_predictions: The card predictions in frames.

        Returns:
            Listing of pulled Pokémon cards
        """

        # obtain reference
        if card_predictions.reference_set is not None:
            reference = ReferenceBuild.get(card_predictions.reference_set)
        else:
            raise ValueError("Unspecified reference.")

        # obtain run card summary
        unique_card_names = [
            reference.cards[run.card_index].name
            + " (#"
            + str(reference.cards[run.card_index].number)
            + ") ["
            + str(run.interval)
            + "]"
            for run in card_predictions.runs
        ]

        # write row to summary file (if specified)
        if self.summary_file is not None:
            if not os.path.exists(self.summary_file):
                first_write = True
            else:
                first_write = False
            with open(self.summary_file, "a") as file_out:
                if first_write:
                    header_cols = ["input_path"]
                    header_cols.extend(
                        ["P_" + str(i + 1) for i in range(len(unique_card_names))]
                    )
                    header = "\t".join(header_cols) + "\n"
                    file_out.write(header)
                line = str(card_predictions.input_path) + "\t"
                line += "\t".join(unique_card_names) + "\n"
                file_out.write(line)

        # return
        return unique_card_names
