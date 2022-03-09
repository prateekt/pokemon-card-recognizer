import os
import pickle

from card_recognizer.classifier.rules import (
    classify_shared_words,
    classify_shared_words_rarity,
    classify_l1,
)
from card_recognizer.classifier.word_classifier import WordClassifier
from card_recognizer.eval.eval import compute_acc_exclude_alt_art
from card_recognizer.ocr.pipeline.framework.ocr_fusion import OCRFusion


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
        images_path = os.path.join(out_folder, "card_images", set_prefix)
        ref_pkl_path = os.path.join(out_folder, "ref_build", set_prefix + ".pkl")
        ocr_result_path = os.path.join(out_folder, "ref_ocr", set_prefix + "_ocr.pkl")

        # init classifier
        classifier = WordClassifier(
            ref_pkl_file=ref_pkl_path, vect_method="encapsulation_match"
        )

        # load OCR results
        if recompute_ocr or not os.path.exists(ocr_result_path):
            os.makedirs(os.path.join(out_folder, "ref_ocr"), exist_ok=True)
            ocr_pipeline = OCRFusion(vocab=classifier.vocab)
            ocr_words = ocr_pipeline.run_on_images(images_dir=images_path)
            pickle.dump(ocr_words, open(ocr_result_path, "wb"))
        else:
            ocr_words = pickle.load(open(ocr_result_path, "rb"))

        # test various classifier rules and print out results
        acc_results = list()
        for classifier_rule in [
            classify_l1,
            classify_shared_words,
            classify_shared_words_rarity,
        ]:
            # make predictions
            preds, scores, all_scores = classifier.classify(
                ocr_words=ocr_words, classification_func=classifier_rule
            )

            # compute accuracy
            acc, incorrect = compute_acc_exclude_alt_art(
                preds=preds,
                gt=range(len(ocr_words)),
                cards_reference=classifier.cards,
            )
            acc_results.append(classifier_rule.__name__ + ": " + str(acc))
        print(set_name + ": " + str(acc_results))


if __name__ == "__main__":
    main()
