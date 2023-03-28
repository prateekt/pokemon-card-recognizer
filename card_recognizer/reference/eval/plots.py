from typing import Dict
from typing import Optional, List

import ezplotly as ep
import ezplotly_bio as epb
import pandas as pd
from ezplotly import EZPlotlyPlot
from scipy.stats import entropy

from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.classifier.eval.sensitivity import (
    compute_classification_sensitivity,
)
from card_recognizer.reference.core.card_reference import CardReference


def plot_word_entropies(
    reference: CardReference, outfile: Optional[str] = None
) -> None:
    """
    Plot word entropies for a particular card reference.

    param reference: The card reference
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

    param references: Dict mapping reference name -> CardReference object
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
            min_bin=0.0,
            max_bin=100.0,
            bin_size=2.0,
            ylim=[0.0, 0.2],
        )
    ep.plot_all(
        h,
        panels=[1] * len(references.keys()),
        showlegend=True,
        outfile=outfile,
        height=500,
        width=800,
        paging={"num_plots_per_subplot": 10, "num_subplots_per_page": 1},
    )


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
        pkl_path = set_pkl_paths[set_name]
        classifier = WordClassifier(
            ref_pkl_path=pkl_path, classification_method=classifier_method
        )
        num_words, accs = compute_classification_sensitivity(
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
        height=500,
        paging={"num_plots_per_subplot": 10, "num_subplots_per_page": 1},
    )


def plot_classifier_rules_performance(
    tbl: pd.DataFrame, outfile: Optional[str] = None
) -> None:
    """
    Plot performance of classifier rules from evaluation table.

    param tbl: Evaluation data table
    param outfile: Path to output file
    """
    num_card_sets = len(tbl.index)
    num_sets_per_page = 10
    for page_index in range(0, num_card_sets, num_sets_per_page):
        h: List[Optional[EZPlotlyPlot]] = [None] * len(tbl.columns)
        for i in range(len(h)):
            h[i] = ep.bar(
                x=tbl.index.values[page_index : page_index + num_sets_per_page],
                y=tbl[tbl.columns[i]][page_index : page_index + num_sets_per_page],
                xlabel="Pokemon Set",
                ylabel="Accuracy",
                name=tbl.columns[i],
                text=[
                    str(round(a, 2))
                    for a in tbl[tbl.columns[i]].values[
                        page_index : page_index + num_sets_per_page
                    ]
                ],
                ylim=[0, 1.0],
                y_dtick=0.25,
                title="Performance of Classifier Rules",
            )
        if outfile is not None:
            outfile_ext = "." + outfile.split(".")[-1]
            new_outfile = outfile.replace(
                outfile_ext, f"_{int(page_index/10)}" + outfile_ext
            )
        else:
            new_outfile = None
        ep.plot_all(
            h,
            panels=[1] * len(h),
            showlegend=True,
            outfile=new_outfile,
        )
