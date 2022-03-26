from typing import Optional, List, Sequence


class CardPrediction:
    """
    A light-weight data structure that represents a single card prediction of a single image frame by a classifier.
    """

    def __init__(self, card_index_in_reference: int, conf: float):
        self.card_index_in_reference: int = card_index_in_reference
        self.conf: float = conf
        self.frame_index: Optional[int] = None
        self.all_probs: Optional[Sequence[float]] = None
        self.reference_set: Optional[str] = None


class CardPredictionResult:
    """
    A light-weight data structure that represents the result of a card prediction task by a classifier.
    The result is typically a list of card predictions.
    """

    def __init__(self, predictions: List[CardPrediction]):
        self.predictions: List[CardPrediction] = predictions
        self.input_path: Optional[str] = None

    def __getitem__(self, i):
        return self.predictions[i]

    def __len__(self):
        return len(self.predictions)
