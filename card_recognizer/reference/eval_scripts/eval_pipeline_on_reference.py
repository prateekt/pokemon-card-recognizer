import functools
import os
from typing import List, Tuple

from natsort import natsorted
from pokemontcgsdk import Card

from card_recognizer.api.card_recognizer_pipeline import CardRecognizerPipeline
from card_recognizer.eval.eval import (
    compute_acc_exclude_alt_art,
    is_correct_exclude_alt_art,
)
from card_recognizer.reference.core.build import ReferenceBuild


def eval_prediction(
    card_reference: List[Card], card_files: List[str], inp: str, pred: Tuple[int, float]
) -> bool:
    gt_card_num = card_files.index(os.path.basename(inp))
    pred = pred[0]
    return is_correct_exclude_alt_art(
        pred=pred, gt=gt_card_num, cards_reference=card_reference
    )


def main():

    # loop
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
        acc_results: List[str] = list()
        for classifier_rule in ["l1", "shared_words", "shared_words_rarity"]:

            # init pipeline
            pipeline = CardRecognizerPipeline(
                set_name=set_name, classification_method=classifier_rule
            )
            card_files = [
                os.path.basename(card.images.large)
                for card in pipeline.classifier.cards
            ]
            eval_prediction_func = functools.partial(
                eval_prediction, pipeline.classifier.cards, card_files
            )
            eval_results = pipeline.evaluate(
                inputs=input_files,
                eval_func=eval_prediction_func,
                incorrect_pkl_path="incorrect_reference_preds",
                mechanism="sequential",
            )
            preds = [result[0][0] for result in eval_results]
            acc, incorrect = compute_acc_exclude_alt_art(
                preds=preds,
                gt=range(len(eval_results)),
                cards_reference=pipeline.classifier.cards,
            )
            acc_results.append(classifier_rule + ": " + str(acc))
        print(set_name + ": " + str(acc_results))


if __name__ == "__main__":
    main()
