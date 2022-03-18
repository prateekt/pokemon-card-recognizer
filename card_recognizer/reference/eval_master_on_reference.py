import functools
import os
from typing import List, Tuple

from natsort import natsorted
from pokemontcgsdk import Card

from card_recognizer.api.card_recognizer_pipeline import CardRecognizerPipeline
from card_recognizer.eval.eval import (
    compute_acc_exclude_alt_art,
    _is_correct_exclude_alt_art,
)


def eval_prediction(
    card_reference: List[Card],
    set_name: str,
    card_files: List[str],
    inp: str,
    pred: Tuple[int, float],
) -> bool:
    gt_card_num = card_files.index(os.path.join(set_name, os.path.basename(inp)))
    pred = pred[0]
    return _is_correct_exclude_alt_art(
        pred=pred, gt=gt_card_num, cards_reference=card_reference
    )


def correct_set_name(proposed_set_name: str) -> str:
    if proposed_set_name == "Brilliant Stars Trainer Gallery":
        return "Brilliant Stars"
    else:
        return proposed_set_name


def main():
    # flags
    out_folder = "data"
    card_sets = [
        "Vivid Voltage",
        "Darkness Ablaze",
        "Chilling Reign",
        "Evolving Skies",
        "Fusion Strike",
        "Brilliant Stars",
    ]

    # loop
    for set_name in card_sets:

        # define paths
        set_prefix = set_name.lower().replace(" ", "_")
        images_path = os.path.join(out_folder, "card_images", set_prefix)
        input_files = natsorted(
            [os.path.join(images_path, file) for file in os.listdir(images_path)]
        )
        ref_pkl_path = os.path.join(out_folder, "ref_build", "master.pkl")

        # test different classifier rules
        acc_results: List[str] = list()
        for classifier_rule in ["l1", "shared_words", "shared_words_rarity"]:
            # init pipeline
            pipeline = CardRecognizerPipeline(ref_pkl_path=ref_pkl_path)
            card_files = [
                os.path.join(
                    correct_set_name(card.set.name), os.path.basename(card.images.large)
                )
                for card in pipeline.classifier.cards
            ]
            gt = [
                card_files.index(os.path.join(set_name, os.path.basename(input_file)))
                for input_file in input_files
            ]
            eval_prediction_func = functools.partial(
                eval_prediction, pipeline.classifier.cards, set_name, card_files
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
                gt=gt,
                cards_reference=pipeline.classifier.cards,
            )
            acc_results.append(classifier_rule + ": " + str(acc))
        print(set_name + ": " + str(acc_results))


if __name__ == "__main__":
    main()
