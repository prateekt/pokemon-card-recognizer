import functools
import os
from typing import List, Optional

import pandas as pd
from natsort import natsorted
from pokemontcgsdk import Card

from card_recognizer.api.card_recognizer import CardRecognizer
from card_recognizer.classifier.core.card_prediction_result import (
    CardPrediction,
    CardPredictionResult,
)
from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.eval.eval import (
    compute_acc_exclude_alt_art,
    is_correct_exclude_alt_art,
)
from card_recognizer.reference.core.build import ReferenceBuild
from card_recognizer.reference.eval.plots import plot_classifier_rules_performance


def _eval_prediction(
    card_reference: List[Card],
    set_name: str,
    card_files: List[str],
    inp: str,
    pred: CardPredictionResult,
) -> bool:
    """
    Helper function for evaluating predictions.
    """
    gt_card_num = card_files.index(os.path.join(set_name, os.path.basename(inp)))
    assert isinstance(pred, CardPredictionResult)
    assert len(pred) == 1
    card_pred = pred[0]
    assert isinstance(card_pred, CardPrediction)
    return is_correct_exclude_alt_art(
        pred=card_pred.card_index_in_reference,
        gt=gt_card_num,
        cards_reference=card_reference,
    )


def _correct_set_name(proposed_set_name: str) -> str:
    """
    Helper function to identify correct set name.
    """
    if proposed_set_name == "Brilliant Stars Trainer Gallery":
        return "Brilliant Stars"
    else:
        return proposed_set_name


def main():
    """
    Script to evaluate master model accuracy on all set card images. Reports an accuracy per set and per rule.
    """

    # create results sample_data frame
    results_df = pd.DataFrame(
        {rule: [] for rule in WordClassifier.get_supported_classifier_methods()}
    )
    for set_name in ReferenceBuild.supported_card_sets():

        # define paths
        set_prefix = set_name.lower().replace(" ", "_")
        images_path = os.path.join(
            ReferenceBuild.get_path_to_data(), "card_images", set_prefix
        )
        input_files = natsorted(
            [os.path.join(images_path, file) for file in os.listdir(images_path)]
        )

        # test different classifier rules
        acc_results: List[float] = list()
        for classifier_rule in WordClassifier.get_supported_classifier_methods():
            # init pipeline
            pipeline = CardRecognizer(
                set_name="master", classification_method=classifier_rule
            )
            card_files = [
                os.path.join(
                    _correct_set_name(card.set.name),
                    os.path.basename(card.images.large),
                )
                for card in pipeline.classifier.reference.cards
            ]
            gt = [
                card_files.index(os.path.join(set_name, os.path.basename(input_file)))
                for input_file in input_files
            ]
            eval_prediction_func = functools.partial(
                _eval_prediction,
                pipeline.classifier.reference.cards,
                set_name,
                card_files,
            )
            eval_result = pipeline.evaluate(
                inputs=input_files,
                eval_func=eval_prediction_func,
                incorrect_pkl_path=os.path.join(
                    ReferenceBuild.get_path_to_data(),
                    "incorrect_master_reference_preds",
                ),
                mechanism="sequential",
            )
            preds: List[Optional[int]] = [None for _ in range(len(input_files))]
            for i, ev_result in enumerate(eval_result):
                result = ev_result[0]
                if result is not None:
                    assert isinstance(result, CardPredictionResult)
                    assert len(result) == 1
                    card_pred = result[0]
                    preds[i] = card_pred.card_index_in_reference
            acc, incorrect = compute_acc_exclude_alt_art(
                preds=preds,
                gt=gt,
                cards_reference=pipeline.classifier.reference.cards,
            )
            acc_results.append(acc)
        results_df.loc[set_name] = acc_results

    # output results sample_data frame to file and plot figure
    results_file_path = os.path.join(
        ReferenceBuild.get_path_to_data(),
        "eval_figs",
        "acc_master_model_on_reference.tsv",
    )
    results_fig_path = os.path.join(
        ReferenceBuild.get_path_to_data(),
        "eval_figs",
        "acc_master_model_on_reference.png",
    )
    results_df.to_csv(results_file_path, sep="\t")
    plot_classifier_rules_performance(tbl=results_df, outfile=results_fig_path)


if __name__ == "__main__":
    main()
