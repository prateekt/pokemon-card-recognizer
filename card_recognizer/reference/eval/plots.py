import random
from typing import Dict
from typing import Tuple, Optional, List

import ezplotly as ep
import ezplotly_bio as epb
import numpy as np
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

    param outfile: File output figure to
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


def _compute_classification_sensitivity(
    classifier: WordClassifier, num_trials: int
) -> Tuple[np.array, np.array]:
    """ """

    # compute acc by num words curve
    ref = classifier.reference.ref_mat
    vocab = classifier.reference.vocab
    cards = classifier.reference.cards
    num_words = np.array([nw for nw in range(1, 80)], dtype=int)
    accs = np.zeros((len(num_words),), dtype=float)
    for i, nw in enumerate(num_words):
        num_correct = 0
        for trial in range(num_trials):

            # generate data
            correct_answer = random.randint(0, ref.shape[0] - 1)
            words_poss = np.where(ref[correct_answer, :] > 0)[0]
            card_words = list()
            for word_poss_index in words_poss:
                card_words.extend(
                    [
                        vocab.inv(word_poss_index)
                        for _ in range(ref[correct_answer, word_poss_index])
                    ]
                )
            draw = nw
            if draw > len(card_words):
                draw = len(card_words)
            random_words = np.random.choice(card_words, draw, replace=False).tolist()
            pred, _ = classifier.classify(ocr_words=random_words)
            if is_correct_exclude_alt_art(
                pred=pred, gt=correct_answer, cards_reference=cards
            ):
                num_correct += 1
        accs[i] = (num_correct / num_trials) * 100.0
    return num_words, accs


def plot_classifier_sensitivity_curve(
    set_pkl_paths: Dict[str, str],
    classifier_method: str,
    num_trials: int = 100,
    outfile: Optional[str] = None,
) -> None:
    """
    Plot sensitivity curves for classifier method over all card sets.
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
            xlabel="# of Randomly Drawn Card Words",
            ylabel="Test Accuracy",
            y_dtick=10,
            title=classifier_method,
        )
    ep.plot_all(
        h, panels=[1] * len(h), showlegend=True, outfile=outfile, suppress_output=True
    )
