from typing import Dict


class Interval:
    """
    Represents the interval [start, end).
    """

    def __init__(self, start: int, end: int):
        """
        param start: Start (inclusive) of interval
        param end: End (exclusive) of interval
        """

        if start < 0:
            raise ValueError("Range start must be >= 0.")
        if end < 0:
            raise ValueError("Range end must be >= 0.")
        if start > end:
            raise ValueError("Range start <= end.")
        self.start = start
        self.end = end

    def __eq__(self, other):
        return other.start == self.start and other.end == self.end

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return str(self.start) + "-" + str(self.end)


class Run:
    """
    Represents a run of frames of a particular card (referenced by card_index). The run spans a particular interval
    in a video or image stream. Each frame has a confidence score. The selection score is the integral of the
    confidence scores.
    """

    def __init__(
        self, interval: Interval, card_index: int, confidence_scores: Dict[int, float]
    ):
        """
        param interval: The interval of the run
        param card_index: The run is of this card index
        confidence_scores: Dict mapping frame number -> confidence score
        """
        self.interval = interval
        self.card_index = card_index
        self.confidence_scores = confidence_scores
        self.max_confidence_score = max(list(confidence_scores.values()))
        self.selection_score = sum(list(confidence_scores.values()))

    def __len__(self):
        return len(self.interval)

    def __lt__(self, other):
        return self.interval.start < other.interval.start
