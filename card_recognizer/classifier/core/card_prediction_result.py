from typing import Optional, List, Sequence, Dict, OrderedDict

from algo_ops.pickleable_object.pickleable_object import PickleableObject

from card_recognizer.run_finding.interval import Run, Interval
from card_recognizer.run_finding.run_finding import find_runs_with_tol, is_sorted


class CardPrediction:
    """
    A light-weight data structure that represents a single card prediction of a single image frame
    by a classifier.
    """

    def __init__(
        self,
        card_index_in_reference: int,
        conf: float,
        frame_index: Optional[int] = None,
    ):
        """
        param card_index_in_reference: The predicted card's index in the reference
        param conf: The confidence score of the prediction
        param frame_index: The frame index (in a video or image stream) of the prediction
        """
        self.card_index_in_reference: int = card_index_in_reference
        self.conf: float = conf
        self.frame_index: Optional[int] = frame_index
        self.all_probs: Optional[Sequence[float]] = None

    def __lt__(self, other):
        if self.frame_index is not None and other.frame_index is not None:
            return self.frame_index < other.frame_index
        else:
            raise ValueError(
                "CardPrediction objects cannot be compared without frame_index"
                "parameter."
            )

    def __le__(self, other):
        if self.frame_index is not None and other.frame_index is not None:
            return self.frame_index <= other.frame_index
        else:
            raise ValueError(
                "CardPrediction objects cannot be compared without frame_index"
                "parameter."
            )

    def __str__(self):
        """
        Human-readable string representation of CardPrediction object.
        """
        if self.frame_index is not None:
            tpl = (
                self.frame_index,
                self.card_index_in_reference,
                self.conf,
            )
        else:
            tpl = (self.card_index_in_reference, self.conf)
        return str(tpl)


class CardPredictionResult(PickleableObject):
    """
    A light-weight sample_data structure that represents the result of a card prediction task by a classifier.
    The result is a list of card predictions, one for each input image.
    """

    def __init__(
        self,
        predictions: List[CardPrediction],
        run_tol: int = 5,
        num_frames: Optional[int] = None,
    ):
        """
        param predictions: The list of card predictions made
        run_tol: When defining runs, the run tolerance to noise parameter
        num_frames: The number of frames processed in the video or image directory (if applicable)
        """
        self.predictions: List[CardPrediction] = predictions
        self.input_path: Optional[str] = None
        self.num_frames: Optional[int] = num_frames
        self.reference_set: Optional[str] = None
        self.unique_cards = self._tabulate_unique_cards()
        if num_frames is not None:
            self.runs = self._find_runs(run_tol=run_tol)
        else:
            self.runs = None

    def __getitem__(self, i):
        return self.predictions[i]

    def __setitem__(self, key, value):
        self.predictions[key] = value

    def __len__(self):
        return len(self.predictions)

    def query_card_prediction(
        self, interval: Interval, card_index: int
    ) -> List[CardPrediction]:
        """
        Query card predictions in interval of a particular card.

        param interval: The frame index interval in time series to query
        param card_index: The card index to look for

        Return:
            List of card predictions of card_index in frame interval
        """
        preds_in_range: List[CardPrediction] = list()
        for card_pred in self.predictions:
            if (
                interval.start <= card_pred.frame_index < interval.end
                and card_pred.card_index_in_reference == card_index
            ):
                preds_in_range.append(card_pred)
            if card_pred.frame_index > interval.end:
                break
        return preds_in_range

    def query_confidence_scores(
        self, interval: Interval, card_index: int
    ) -> Dict[int, float]:
        """
        Query confidence scores in interval of a particular card.

        param range: The frame index range to query
        param card_index: The card index to look for

        Return:
            conf_in_range: Dict mapping frame_index -> conf score
        """
        assert is_sorted(self.predictions)
        conf_in_range: Dict[int, float] = dict()
        for card_pred in self.predictions:
            if card_pred.card_index_in_reference != card_index:
                continue
            if interval.start <= card_pred.frame_index < interval.end:
                conf_in_range[card_pred.frame_index] = card_pred.conf
            if card_pred.frame_index > interval.end:
                break
        return conf_in_range

    def __str__(self):
        """
        Human-readable representation of CardPredictionResult.
        """
        if len(self.predictions) == 0:
            return "Empty Results."
        else:
            return str([str(pred) for pred in self.predictions])

    def to_int_series(self) -> List[Optional[int]]:
        """
        Represents card prediction results as integer series. If no prediction was made for a frame, None is returned.

        Returns:
            List of integers representing complete time series up to num_frames
        """
        if self.num_frames is None:
            raise ValueError("num_frames cannot be None.")
        rtn: List[Optional[int]] = [None for _ in range(self.num_frames)]
        for i, pred in enumerate(self.predictions):
            rtn[pred.frame_index] = pred.card_index_in_reference
        return rtn

    def _tabulate_unique_cards(self) -> List[int]:
        """
        Tabulate unique pulls (in temporal order of detection using ordered dictionary)
        """
        unique_cards: List[int] = list(
            OrderedDict.fromkeys(
                [
                    frame_prediction.card_index_in_reference
                    for frame_prediction in self.predictions
                ]
            ).keys()
        )
        return unique_cards

    def _find_runs(self, run_tol: int) -> List[Run]:
        """
        Finds runs of a card detection in consecutive frames.

        param run_tol: How many frames consecutively can be noise predictions to collapse one run into another.

        Returns:
            List of detected card runs
        """

        # compute runs for each unique card
        runs: List[Run] = list()
        series = self.to_int_series()
        for card_index in self.unique_cards:
            intervals = find_runs_with_tol(
                series=series, query_elem=card_index, tol=run_tol
            )
            for interval in intervals:
                confidence_scores = self.query_confidence_scores(
                    interval=interval, card_index=card_index
                )
                runs += [
                    Run(
                        interval=interval,
                        card_index=card_index,
                        confidence_scores=confidence_scores,
                    )
                ]

        # return sorted runs
        runs.sort()
        return runs
