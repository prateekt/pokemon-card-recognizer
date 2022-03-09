import os
from typing import List, Union

import cv2
import numpy as np
from natsort import natsorted

from card_recognizer.infra.paraloop import paraloop
from card_recognizer.ocr.pipeline.instances import ocr
from card_recognizer.reference.vocab import Vocab


class OCRFusion:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.basic_pytesseract_pipeline = ocr.basic_pytesseract_pipeline()
        self.black_text_ocr_pipeline = ocr.black_text_ocr_pipeline()
        self.white_text_ocr_pipeline = ocr.white_text_ocr_pipeline()
        self.basic_pytesseract_pipeline.set_text_pipeline_params(
            func_name="_check_vocab", params={"vocab": vocab}
        )
        self.black_text_ocr_pipeline.set_text_pipeline_params(
            func_name="_check_vocab", params={"vocab": vocab}
        )
        self.white_text_ocr_pipeline.set_text_pipeline_params(
            func_name="_check_vocab", params={"vocab": vocab}
        )
        self.lower_lim_trials = [0, 50, 90, 100, 150, 190, 200, 210, 220, 250]
        self.vis = False

    def run(
        self,
        img: np.array,
    ) -> List[str]:
        """
        param img: Input image

        return:
            output: List of detected words
        """

        # run defaults
        ocr_words = self.basic_pytesseract_pipeline.run(input_img=img)
        if self.vis:
            print("Basic Pytesseract OCR")
            self.basic_pytesseract_pipeline.vis()
        final_vect = self.vocab.vect(words=ocr_words)

        # loop over hyper parameters
        for lower_lim in self.lower_lim_trials:

            # run black text ocr pipeline on lower lim parameter setting
            self.black_text_ocr_pipeline.set_img_pipeline_params(
                func_name="_remove_background", params={"lower_lim": lower_lim}
            )
            black_text = self.black_text_ocr_pipeline.run(input_img=img)
            black_text_vect = self.vocab.vect(black_text)
            if self.vis:
                print("Black Text OCR (ll=" + str(lower_lim) + ")")
                self.black_text_ocr_pipeline.vis()

            # run white text ocr pipeline on lower lim parameter setting
            self.white_text_ocr_pipeline.set_img_pipeline_params(
                func_name="_remove_background", params={"lower_lim": lower_lim}
            )
            white_text = self.white_text_ocr_pipeline.run(input_img=img)
            white_text_vect = self.vocab.vect(white_text)
            if self.vis:
                print("White Text OCR (ll=" + str(lower_lim) + ")")
                self.white_text_ocr_pipeline.vis()

            # update final vect
            final_vect = np.array(
                [
                    np.max([final_vect[i], black_text_vect[i], white_text_vect[i]])
                    for i in range(len(final_vect))
                ],
                dtype=int,
            )

        # compute final OCR-ed words
        final_ocr_words = [
            self.vocab._words.inv[i] for i in np.where(final_vect > 0)[0]
        ]
        if self.vis:
            print("Final: " + str(final_ocr_words))
        return final_ocr_words

    def run_on_img_file(self, file: str) -> List[str]:
        """
        Runs OCR pipeline on input image file.

        param file: Path to input image file

        return:
            output: List of detected words
        """
        img = cv2.imread(filename=file)
        ocr_words = self.run(img=img)
        return ocr_words

    def run_on_images(self, images_dir: str) -> Union[List[str], List[List[str]]]:
        """
        API to run OCR on a directory of images.

        param files_path: Path to directory of card image files

        return:
            output: List of OCR results
        """
        files = natsorted(
            [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
        )
        results = paraloop.loop(func=self.run_on_img_file, params=files)
        return results
