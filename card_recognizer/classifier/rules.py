from typing import Tuple

import numpy as np


def classify_l1(v: np.array, ref_mat: np.array) -> Tuple[int, float, np.array]:
    """
    Nearest match of word vector (v) to reference matrix based on L1 norm.

    param v: `np.array[int]` Word vector
    param ref_mat: `np.array[int]` Set reference matrix

    return:
        card_number_prediction: The predicted card number
        score: The score of top match
    """
    d = np.sum(np.abs(ref_mat - v), axis=1)
    card_number_prediction = d.argmin()
    score = d[card_number_prediction]
    return card_number_prediction, score, d


def classify_shared_words(
    v: np.array, ref_mat: np.array
) -> Tuple[int, float, np.array]:
    """
    Classifies the nearest match of word vector v based on count of shared words with reference card text.

    param v: `np.array[int]` Word vector
    param ref_mat: `np.array[int]` Set reference matrix

    return:
        card_number_prediction: The predicted card number
        score: The score of top match
    """
    scores = ((ref_mat > 0) & (v > 0)).sum(axis=1)
    card_number_prediction = scores.argmax()
    score = scores[card_number_prediction]
    return card_number_prediction, score, scores


def classify_shared_words_rarity(
    v: np.array, ref_mat: np.array
) -> Tuple[int, float, np.array]:
    """
    Classifies the nearest match of word vector v based on count of shared words with reference card text,
    weighted by rarity of word.

    param v: `np.array[int]` Word vector
    param ref_mat: `np.array[int]` Set reference matrix

    return:
        card_number_prediction: The predicted card number
        score: The score of top match
    """
    rarity = ref_mat / ref_mat.sum(axis=0)
    scores = (rarity * ((ref_mat > 0) & (v > 0))).sum(axis=1)
    card_number_prediction = scores.argmax()
    score = scores[card_number_prediction]
    return card_number_prediction, score, scores
