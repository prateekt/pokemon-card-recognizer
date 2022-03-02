import os
import pickle

from card_recognizer.ocr.ocr import ocr_cards
from card_recognizer.algo.text_classify import classify


def main():
    # params
    set_name = "Chilling Reign"
    set_prefix = set_name.lower().replace(" ", "_")
    images_path = "/Users/tandonp/Desktop/frames"
    ref_pkl_path = os.path.join("../reference", "data", set_prefix + ".pkl")
    ocr_result_path = os.path.join("test_ocr.pkl")
    recompute_ocr = False

    # algo cards
    if recompute_ocr or not os.path.exists(ocr_result_path):
        ocr_results = ocr_cards(files_path=images_path)
        pickle.dump(ocr_results, open(ocr_result_path, "wb"))
    else:
        ocr_results = pickle.load(open(ocr_result_path, "rb"))

    # load reference
    ref, vocab, cards = pickle.load(open(ref_pkl_path, "rb"))

    # make predictions
    preds, scores = classify(ocr=ocr_results, ref_mat=ref, vocab=vocab)
    print(set(preds))


if __name__ == "__main__":
    main()
