import os
from typing import List, Optional

from algo_ops.ops.text import TextOp

from card_recognizer.api.operating_mode import OperatingMode
from card_recognizer.classifier.core.card_frame_run import CardFrameRun
from card_recognizer.classifier.core.card_prediction_result import CardPredictionResult
from card_recognizer.reference.core.build import ReferenceBuild
from card_recognizer.reference.core.card_reference import CardReference


class PullsSummary(TextOp):
    """
    Converts a time series of frame card predictions into a simple summary of pulled cards.
    """

    def __init__(
        self,
        operating_mode: OperatingMode,
        summary_file: Optional[str] = None,
    ):
        """
        param mode: The operating mode of the card recognizer.
        param summary_file: Path to where summary file should be written. If None, no summary file is written.
        """

        # set params
        super().__init__(func=self.make_pulls_summary)
        self.operating_mode = operating_mode
        self.summary_file = summary_file

        # define input/output types
        self.input: Optional[CardPredictionResult] = None
        self.output: Optional[List[str]] = None

    def _format_pull_display_str(
        self, run: CardFrameRun, reference: CardReference
    ) -> str:
        """
        Formats a pull display string for a single pull.

        param run: The run to format
        param reference: The reference build

        Returns:
            Formatted pull string
        """

        # obtain pull info
        card_name = str(reference.cards[run.card_index].name)
        card_number = str(reference.cards[run.card_index].number)
        run_interval = str(run.interval)

        if self.operating_mode != OperatingMode.SINGLE_IMAGE:
            run_str = " [" + run_interval + "]"
        else:
            run_str = ""

        # format pull string
        if reference.name == "master":
            set_name = reference.cards[run.card_index].set.name
            return card_name + " (" + set_name + " #" + card_number + ")" + run_str
        else:
            return card_name + " (#" + card_number + ")" + run_str

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

        # obtain run card summary as a list of strings
        if len(card_predictions.runs) != 0:
            unique_card_names = [
                self._format_pull_display_str(run=run, reference=reference)
                for run in card_predictions.runs
            ]
        else:
            unique_card_names = []

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
