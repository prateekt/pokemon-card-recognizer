import functools
import os
from typing import List, Optional

import pandas as pd
from natsort import natsorted
from pokemontcgsdk import Card

from card_recognizer.api.card_recognizer_pipeline import CardRecognizerPipeline
from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
)
from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.eval.eval import (
    compute_acc_exclude_alt_art,
    is_correct_exclude_alt_art,
)
from card_recognizer.reference.core.build import ReferenceBuild


def _eval_prediction(
    card_reference: List[Card],
    card_files: List[str],
    inp: str,
    pred: CardPredictionResult,
) -> bool:
    gt_card_num = card_files.index(os.path.basename(inp))
    assert isinstance(pred, CardPredictionResult)
    assert len(pred) == 1
    card_pred = pred[0]
    assert isinstance(card_pred, CardPrediction)
    return is_correct_exclude_alt_art(
        pred=card_pred.card_index_in_reference,
        gt=gt_card_num,
        cards_reference=card_reference,
    )


def main():
    """
    Script to evaluate each individual set model accuracy on all set card images. Reports an accuracy per set and per
    rule for each model tested on the correct set.
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
        for classifier_rule in ["l1", "shared_words", "shared_words_rarity"]:

            # init pipeline
            pipeline = CardRecognizerPipeline(
                set_name=set_name, classification_method=classifier_rule
            )
            card_files = [
                os.path.basename(card.images.large)
                for card in pipeline.classifier.reference.cards
            ]
            eval_prediction_func = functools.partial(
                _eval_prediction, pipeline.classifier.reference.cards, card_files
            )
            eval_result, _ = pipeline.evaluate(
                inputs=input_files,
                eval_func=eval_prediction_func,
                incorrect_pkl_path=os.path.join(
                    ReferenceBuild.get_path_to_data(), "incorrect_set_reference_preds"
                ),
                mechanism="sequential",
            )
            preds: List[Optional[int]] = [None for _ in range(len(input_files))]
            for i, result in enumerate(eval_result):
                if result is not None:
                    assert isinstance(result, CardPredictionResult)
                    assert len(result) == 1
                    card_pred = result[0]
                    preds[i] = card_pred.card_index_in_reference
            acc, incorrect = compute_acc_exclude_alt_art(
                preds=preds,
                gt=range(len(input_files)),
                cards_reference=pipeline.classifier.reference.cards,
            )
            acc_results.append(acc)
        results_df.loc[set_name] = acc_results

    # output results sample_data frame to file
    results_file_path = os.path.join(
        ReferenceBuild.get_path_to_data(), "eval_figs", "acc_set_model_on_reference.tsv"
    )
    results_df.to_csv(results_file_path, sep="\t")


if __name__ == "__main__":
    main()
