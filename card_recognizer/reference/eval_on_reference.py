import os
import pickle

from card_recognizer.eval.eval import compute_acc_exclude_alt_art
from card_recognizer.algo.ocr import ocr_cards
from card_recognizer.algo.text_classify import classify


def main():
    out_folder = 'data'
    card_sets = ['Vivid Voltage', 'Darkness Ablaze', 'Chilling Reign', 'Evolving Skies', 'Fusion Strike',
                 'Brilliant Stars']
    recompute_ocr = False
    for set_name in card_sets:

        # define paths
        set_prefix = set_name.lower().replace(' ', '_')
        images_path = os.path.join(out_folder, set_prefix)
        ref_pkl_path = os.path.join(out_folder, set_prefix + '.pkl')
        ocr_result_path = os.path.join(out_folder, set_prefix + '_ocr.pkl')

        # algo cards
        if recompute_ocr or not os.path.exists(ocr_result_path):
            ocr_results = ocr_cards(files_path=images_path)
            pickle.dump(ocr_results, open(ocr_result_path, 'wb'))
        else:
            ocr_results = pickle.load(open(ocr_result_path, 'rb'))

        # load reference
        ref, vocab, cards = pickle.load(open(ref_pkl_path, 'rb'))

        # make predictions
        preds, scores = classify(ocr=ocr_results, ref_mat=ref, vocab=vocab)

        # compute accuracy
        acc, _ = compute_acc_exclude_alt_art(preds=preds, gt=range(len(ocr_results)),
                                             cards_reference=cards)
        print(set_name + ': ' + str(acc))


if __name__ == "__main__":
    main()
