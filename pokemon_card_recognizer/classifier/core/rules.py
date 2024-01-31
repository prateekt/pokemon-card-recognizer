from typing import Tuple

import numpy as np


def _norm_distribution(d: np.array) -> np.array:
    """
    Normalize distance measure measurements into probability distribution.

    param d: Distance measurements

    return:
        prob: Normalized probability distribution
    """
    min_val = np.min(d)
    max_val = np.max(d)
    rng = max_val - min_val
    if rng > 0:
        prob = (d - min_val) / (max_val - min_val)
    else:
        prob = d  # all zeroes (so prob is also all zeros)
    return prob


def _to_portions(v: np.array) -> np.array:
    """
    Normalize vector to portion vector.

    param v: Input vector

    return:
        Normalized vector
    """
    v_sum = np.sum(v)
    if v_sum > 0:
        v_portions = v / v_sum
    else:
        v_portions = v
    return v_portions


def classify_l1(word_portions_mat: np.array, v: np.array) -> Tuple[int, np.array]:
    """
    Nearest match of word vector (v) to reference matrix based on L1 norm.

    param word_portions_mat: `np.array[int]` Set Word Portions matrix
    param v: `np.array[int]` Word vector

    return:
        card_number_prediction: The predicted card number
        prob: The posterior probabilities for each card
    """
    v_portions = _to_portions(v=v)
    d = np.sum(np.abs(word_portions_mat - v_portions), axis=1)
    prob = 1.0 - _norm_distribution(d=d)
    card_number_prediction = prob.argmax()
    return card_number_prediction, prob


def classify_shared_words(ref_mat: np.array, v: np.array) -> Tuple[int, np.array]:
    """
    Classifies the nearest match of word vector v based on count of shared words with reference card text.

    param ref_mat: `np.array[int]` Set reference matrix
    param v: `np.array[int]` Word vector

    return:
        card_number_prediction: The predicted card number
        prob: The posterior probabilities for each card
    """
    scores = ((ref_mat > 0) & (v > 0)).sum(axis=1)
    prob = scores / (ref_mat > 0).sum(axis=1)
    card_number_prediction = scores.argmax()
    return card_number_prediction, prob


def classify_shared_words_rarity(
    rarity_mat: np.array,
    v: np.array,
) -> Tuple[int, np.array]:
    """
    Classifies the nearest match of word vector v based on count of shared words with reference card text,
    weighted by rarity of word.

    param v: `np.array[int]` Word vector
    param rarity_mat: `np.array[int]` Set rarity matrix

    return:
        card_number_prediction: The predicted card number
        prob: The posterior probabilities for each card
    """
    d = (rarity_mat * ((rarity_mat > 0) & (v > 0))).sum(axis=1)
    prob = _norm_distribution(d=d)
    card_number_prediction = prob.argmax()
    return card_number_prediction, prob
