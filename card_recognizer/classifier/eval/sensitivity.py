import functools
import random
from typing import Tuple, List

import numpy as np
from algo_ops.paraloop import paraloop

from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.eval.eval import is_correct_exclude_alt_art


def _run_classifier_trial(
    classifier: WordClassifier,
    num_words_to_draw: int,
    trial_num: int,
) -> bool:
    """
    Runs a classifier trial where a random card is drawn, random words from the card are drawn, the classifier is run
    on the set of random words, and the prediction is compared to the ground truth. Returns true if the prediction was
    correct.

    param classifier: The classifier
    param num_words_to_draw: The number of words to draw from a randomly drawn card
    param trial_num: The trial number

    Returns:
        True if classifier prediction was correct on randomly generated sample_data
    """

    # extract meta sample_data pointers
    assert isinstance(trial_num, int)
    ref = classifier.reference.ref_mat
    vocab = classifier.reference.vocab

    # draw a card and obtain correct words
    correct_answer = random.randint(0, ref.shape[0] - 1)
    words_poss = np.where(ref[correct_answer, :] > 0)[0]
    card_words: List[str] = list()
    for word_poss_index in words_poss:
        card_words.extend(
            [vocab.inv(word_poss_index)] * ref[correct_answer, word_poss_index]
        )

    # draw random words from card
    if num_words_to_draw > len(card_words):
        num_words_to_draw = len(card_words)
    random_words = np.random.choice(
        card_words, num_words_to_draw, replace=False
    ).tolist()

    # predict and evaluate
    pred = classifier.classify(ocr_words=random_words)
    if is_correct_exclude_alt_art(
        pred=pred[0].card_index_in_reference,
        gt=correct_answer,
        cards_reference=classifier.reference.cards,
    ):
        return True
    else:
        return False


def compute_classification_sensitivity(
    classifier: WordClassifier, num_trials: int, num_max_words: int = 100
) -> Tuple[np.array, np.array]:
    """
    Computes an accuracy by num random words curve. Used to test sensitivity of classifier. How many words does the
    OCR machine need to get right on a card to predict the card right?

    param classifier: The classifier
    num_trials: The number of random card trials to do
    num_max_words: How many word trials to do

    Returns:
        num_words: Sequence of word number trials
        accs: The accuracy at each word number of randomly generated words
    """
    # compute acc by num random words curve
    num_words = np.array([nw for nw in range(1, num_max_words + 1)], dtype=int)
    accs = np.zeros((len(num_words),), dtype=float)
    for i, nw in enumerate(num_words):
        trial_func = functools.partial(_run_classifier_trial, classifier, nw)
        results = paraloop.loop(
            func=trial_func, params=list(range(num_trials)), mechanism="sequential"
        )
        accs[i] = np.sum(results) / num_trials * 100.0
    return num_words, accs
