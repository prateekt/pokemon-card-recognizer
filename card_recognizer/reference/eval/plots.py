import functools
import random
from typing import Dict
from typing import Tuple, Optional, List

import ezplotly as ep
import ezplotly_bio as epb
import numpy as np
from algo_ops.paraloop import paraloop
from ezplotly import EZPlotlyPlot
from scipy.stats import entropy

from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.eval.eval import is_correct_exclude_alt_art
from card_recognizer.reference.core.card_reference import CardReference


def plot_word_entropies(
    reference: CardReference, outfile: Optional[str] = None
) -> None:
    """
    Plot world entropies for a particular card reference.

    param set_name: The name of the set
    outfile: File to output figure to
    """
    word_probs = reference.ref_mat / reference.ref_mat.sum(axis=0)
    entropies = [entropy(word_probs[:, i], base=2) for i in range(word_probs.shape[0])]
    epb.rcdf(
        entropies,
        xlabel="Entropy",
        norm=True,
        y_dtick=0.1,
        outfile=outfile,
        title=reference.name + ": Word Entropies",
    )


def plot_word_counts(
    references: Dict[str, CardReference], outfile: Optional[str] = None
) -> None:
    """
    Plot word count distribution for sets.

    param outfile: Path to file output figure
    """
    h: List[Optional[EZPlotlyPlot]] = [None] * len(references.keys())
    for i, set_name in enumerate(references.keys()):
        words_per_card = references[set_name].ref_mat.sum(axis=1)
        h[i] = ep.hist(
            words_per_card,
            xlabel="# of Words in Card",
            name=set_name,
            title="# of Words Per Card",
            histnorm="probability",
        )
    ep.plot_all(
        h,
        panels=[1] * len(references.keys()),
        showlegend=True,
        outfile=outfile,
        suppress_output=True,
    )


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
        True if classifier prediction was correct on randomly generated data
    """

    # extract meta data pointers
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
    pred, _ = classifier.classify(ocr_words=random_words)
    if is_correct_exclude_alt_art(
        pred=pred, gt=correct_answer, cards_reference=classifier.reference.cards
    ):
        return True
    else:
        return False


def _compute_classification_sensitivity(
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


def plot_classifier_sensitivity_curve(
    set_pkl_paths: Dict[str, str],
    classifier_method: str,
    num_trials: int = 100,
    outfile: Optional[str] = None,
) -> None:
    """
    Plot sensitivity curves for classifier method over all card sets.

    param set_pkl_path: Dict mapping set name to set pkl path
    classifier_method: The classifier method to use
    num_trials: The number of trials over each word list length
    outfile: Path to output figure file
    """
    h: List[Optional[EZPlotlyPlot]] = [None] * len(set_pkl_paths.keys())
    for i, set_name in enumerate(set_pkl_paths.keys()):
        # load classifier for set
        pkl_path = set_pkl_paths[set_name]
        classifier = WordClassifier(
            ref_pkl_path=pkl_path, classification_method=classifier_method
        )
        num_words, accs = _compute_classification_sensitivity(
            classifier=classifier, num_trials=num_trials
        )
        h[i] = ep.line(
            num_words,
            accs,
            name=set_name,
            xlabel="Number of Randomly Drawn Card Words",
            ylabel="Test Accuracy",
            y_dtick=10,
            title=classifier_method,
        )
    ep.plot_all(
        h,
        panels=[1] * len(h),
        showlegend=True,
        outfile=outfile,
        suppress_output=True,
        height=500,
    )
