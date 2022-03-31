from collections import namedtuple

# NamedTuple container for computed statistics
# Has to be put in its own file to avoid circular dependencies.
PullStats = namedtuple(
    "PullStats",
    [
        "unique_cards",
        "card_frequencies",
        "confidence_scores",
        "max_confidence_scores",
        "selection_scores",
        "frame_card_predictions",
        "reference",
    ],
)
