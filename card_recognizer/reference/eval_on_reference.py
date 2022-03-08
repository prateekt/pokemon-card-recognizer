import os
import pickle

from card_recognizer.classifier.word_classifier import WordClassifier
from card_recognizer.eval.eval import compute_acc_exclude_alt_art
from card_recognizer.ocr.ocr import ocr_cards
from card_recognizer.classifier.rules import (
    classify_shared_words,
    classify_shared_words_rarity,
    classify_l1,
)
from card_recognizer.ocr.pipeline.instances.text import basic_text_cleaning_pipeline
from card_recognizer.infra.paraloop import paraloop as paraloop


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
    recompute_ocr = False

    # loop
    for set_name in card_sets:

        # define paths
        set_prefix = set_name.lower().replace(" ", "_")
        images_path = os.path.join(out_folder, set_prefix)
        ref_pkl_path = os.path.join(out_folder, set_prefix + ".pkl")
        ocr_result_path = os.path.join(out_folder, set_prefix + "_ocr.pkl")

        # load OCR results
        if recompute_ocr or not os.path.exists(ocr_result_path):
            ocr_results = ocr_cards(files_path=images_path)
            pickle.dump(ocr_results, open(ocr_result_path, "wb"))
        else:
            ocr_results = pickle.load(open(ocr_result_path, "rb"))

        # init classifier
        classifier = WordClassifier(
            ref_pkl_file=ref_pkl_path, vect_method="encapsulation_match"
        )

        # tokenize and clean raw OCR results
        text_cleaning_pipeline = basic_text_cleaning_pipeline()
        text_cleaning_pipeline.set_params(func_name='_check_vocab', params={'vocab': classifier.vocab})
        ocr_words = paraloop.loop(func=text_cleaning_pipeline.run, params=ocr_results)

        # test various classifier rules and print out results
        acc_results = list()
        for classifier_rule in [classify_l1, classify_shared_words, classify_shared_words_rarity]:
            # make predictions
            preds, scores, all_scores = classifier.classify(
                ocr_words=ocr_words, classification_func=classifier_rule
            )

            # compute accuracy
            acc, incorrect = compute_acc_exclude_alt_art(
                preds=preds, gt=range(len(ocr_results)), cards_reference=classifier.cards
            )
            acc_results.append(classifier_rule.__name__ + ': ' + str(acc))
        print(set_name + ": " + str(acc_results))


if __name__ == "__main__":
    main()
