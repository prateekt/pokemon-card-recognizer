from typing import List, Tuple, Sequence, Optional

import numpy as np
from pokemontcgsdk import Card


def compute_basic_acc(preds: np.array, gt: np.array) -> [float, List[int]]:
    """
    Computes accuracy, excluding alternate art duplicates.

    param preds: The predictions made for each card as list of card numbers(np.array[int])
    param gt: The ground truth card number for each point (np.array[int])
    return:
        acc: Computed accuracy
        incorrect: Vector of incorrect prediction indices
    """
    assert len(preds) == len(gt), "preds and gt must have same size."
    acc = np.sum(preds == gt) / len(gt)
    incorrect = np.where(preds != gt)[0]
    return acc, incorrect


def is_correct_exclude_alt_art(pred: int, gt: int, cards_reference: List[Card]) -> bool:
    """
    Helper to determine if a prediction is correct or same as an alternate art.
    """
    return (pred is not None) and (
        (pred == gt) | (cards_reference[pred].name == cards_reference[gt].name)
    )


def compute_acc_exclude_alt_art(
    preds: Sequence[Optional[int]], gt: Sequence[int], cards_reference: List[Card]
) -> Tuple[float, List[int]]:
    """
    Computes accuracy, excluding alternate art duplicates.

    param preds: The predictions made for each card as list of card numbers(np.array[int])
    param gt: The ground truth card number for each point (np.array[int])
    param cards_reference: Cards reference for set
    return:
        acc: Computed accuracy
    """
    assert len(preds) == len(gt), "preds and gt must have same size."
    num_correct = 0
    incorrect = np.full((len(preds),), True, dtype=bool)
    for i, pred in enumerate(preds):
        if is_correct_exclude_alt_art(
            pred=pred, gt=gt[i], cards_reference=cards_reference
        ):
            incorrect[i] = False
            num_correct += 1
    acc = num_correct / len(preds)
    incorrect = np.where(incorrect)[0]
    return acc, incorrect
