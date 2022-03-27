import os
from typing import List, Union, Dict, Any, Set

import cv2
import numpy as np
from algo_ops.ops.op import Op
from algo_ops.paraloop import paraloop
from natsort import natsorted

from card_recognizer.ocr.framework.pipeline.ocr_pipeline import OCRPipeline
from card_recognizer.ocr.instances import ocr


class OCRFusion(Op):
    def save_output(self, out_path: str = ".") -> None:
        pass

    def save_input(self, out_path: str = ".") -> None:
        pass

    def vis(self) -> None:
        pass

    def vis_input(self) -> None:
        pass

    def _run(
        self,
        file: str,
    ) -> List[str]:
        """
        param img: Input image

        return:
            output: List of detected words
        """

        # load img file
        img = cv2.imread(filename=file)

        # run defaults
        ocr_words = self.basic_pytesseract_pipeline.exec(inp=img)
        if self.vis:
            print("Basic Pytesseract OCR")
            self.basic_pytesseract_pipeline.vis()
        final_vect = self.vocab.vect(words=ocr_words)

        # loop over hyper parameters
        for lower_lim in self.lower_lim_trials:

            # run black text ocr pipeline on lower lim parameter setting
            black_text = self.run_param(
                ocr_pipeline=self.black_text_ocr_pipeline,
                func_name="_remove_background",
                params={"lower_lim": lower_lim},
                inp=img,
            )
            black_text_vect = self.vocab.vect(black_text)
            if self.vis:
                print("Black Text OCR (ll=" + str(lower_lim) + ")")
                self.black_text_ocr_pipeline.vis()

            # run white text ocr pipeline on lower lim parameter setting
            white_text = self.run_param(
                ocr_pipeline=self.white_text_ocr_pipeline,
                func_name="_remove_background",
                params={"lower_lim": lower_lim},
                inp=img,
            )
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
        final_ocr_words = [self.vocab.inv(i) for i in np.where(final_vect > 0)[0]]
        if self.vis:
            print("Final: " + str(final_ocr_words))
        return final_ocr_words

    def __init__(self, vocab_words: Set[str]):
        super().__init__(func=self._run)
        self.vocab_words = vocab_words
        self.basic_pytesseract_pipeline = ocr.basic_ocr_with_text_cleaning_pipeline(
            vocab_words=vocab_words
        )
        self.black_text_ocr_pipeline = ocr.black_text_ocr_pipeline()
        self.white_text_ocr_pipeline = ocr.white_text_ocr_pipeline()
        self.basic_pytesseract_pipeline.set_text_pipeline_params(
            func_name="_check_vocab", params={"vocab_words": vocab_words}
        )
        self.black_text_ocr_pipeline.set_text_pipeline_params(
            func_name="_check_vocab", params={"vocab_words": vocab_words}
        )
        self.white_text_ocr_pipeline.set_text_pipeline_params(
            func_name="_check_vocab", params={"vocab_words": vocab_words}
        )
        self.lower_lim_trials = [100, 150, 200, 250]
        self.vis = False

    @staticmethod
    def run_param(
        ocr_pipeline: OCRPipeline, func_name: str, params: Dict[Any, Any], inp: np.array
    ) -> List[str]:
        ocr_pipeline.set_img_pipeline_params(func_name=func_name, params=params)
        result = ocr_pipeline.exec(inp=inp)
        return result

    def vis_profile(self):
        """
        Visualizes each component.
        """
        self.basic_pytesseract_pipeline.vis_profile()
        print()
        self.black_text_ocr_pipeline.vis_profile()
        print()
        self.white_text_ocr_pipeline.vis_profile()
        print()
        super().vis_profile()

    def run_on_images(
        self, images_dir: str, mechanism: str = "pool"
    ) -> Union[List[str], List[List[str]]]:
        """
        API to run OCR on a directory of images.

        param files_path: Path to directory of card image files

        return:
            output: List of OCR results
        """
        files = natsorted(
            [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
        )
        results = paraloop.loop(func=self.exec, params=files, mechanism=mechanism)
        return results
