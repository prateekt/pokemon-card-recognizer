from typing import Optional, List, Sequence

from algo_ops.pickleable_object.pickleable_object import PickleableObject


class CardPrediction:
    """
    A light-weight sample_data structure that represents a single card prediction of a single image frame
    by a classifier.
    """

    def __init__(self, card_index_in_reference: int, conf: float):
        self.card_index_in_reference: int = card_index_in_reference
        self.conf: float = conf
        self.frame_index: Optional[int] = None
        self.all_probs: Optional[Sequence[float]] = None

    def __str__(self):
        """
        Human-readable string representation of Card Prediction object.
        """
        if self.frame_index is not None:
            tpl = (
                str(self.frame_index),
                str(self.card_index_in_reference),
                str(self.conf),
            )
        else:
            tpl = (str(self.card_index_in_reference), str(self.conf))
        return str(tpl)


class CardPredictionResult(PickleableObject):
    """
    A light-weight sample_data structure that represents the result of a card prediction task by a classifier.
    The result is typically a list of card predictions.
    """

    def __init__(self, predictions: List[CardPrediction]):
        self.predictions: List[CardPrediction] = predictions
        self.input_path: Optional[str] = None
        self.num_frames: Optional[int] = None
        self.reference_set: Optional[str] = None

    def __getitem__(self, i):
        return self.predictions[i]

    def __len__(self):
        return len(self.predictions)

    def __str__(self):
        """
        Human-readable representation of CardPredictionResult.
        """
        if len(self.predictions) == 0:
            return "Empty Results."
        else:
            return str([str(pred) for pred in self.predictions])
