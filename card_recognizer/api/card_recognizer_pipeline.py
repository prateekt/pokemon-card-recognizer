import os
import pickle
from typing import Union, List, Tuple, Optional

from natsort import natsorted

from card_recognizer.classifier.word_classifier import WordClassifier
from card_recognizer.infra.algo_ops.ops.text_op import TextOp
from card_recognizer.infra.algo_ops.pipeline.pipeline import Pipeline
from card_recognizer.infra.paraloop import paraloop

# from card_recognizer.ocr.pipeline.framework.ocr_fusion import OCRFusion
from card_recognizer.ocr.pipeline.framework.ocr_op import OCRMethod
from card_recognizer.ocr.pipeline.instances import ocr


class CardRecognizerPipeline(Pipeline):
    def _classify_func(
        self, ocr_words: Union[List[str], List[List[str]]]
    ) -> Tuple[Optional[int], Optional[float]]:
        card_pred, probs = self.classifier.classify(
            ocr_words=ocr_words, include_probs=True
        )
        if card_pred is None:
            return None, None
        return card_pred, probs[card_pred]

    def __init__(self, ref_pkl_path: str):

        # load classifier
        self.classifier = WordClassifier(
            ref_pkl_file=ref_pkl_path, vect_method="encapsulation_match"
        )

        # load OCR pipeline
        #        ocr_pipeline = OCRFusion(vocab=self.classifier.vocab)
        ocr_pipeline = ocr.basic_ocr_with_text_cleaning_pipeline(
            vocab=self.classifier.vocab, ocr_method=OCRMethod.EASYOCR
        )

        # make pipeline
        ops = [ocr_pipeline, TextOp(func=self._classify_func)]
        super().__init__(ops=ops)

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


if __name__ == "__main__":
    ref_pkl_file = (
        "/home/borg1/Desktop/pokemon-card-recognizer/card_recognizer/reference/data/ref_build/brilliant_stars.pkl"
    )
    pipeline = CardRecognizerPipeline(ref_pkl_file)
    r = pipeline.run_on_images("/home/borg1/Desktop/ttframes", mechanism="sequential")
    pickle.dump(r, open("results_frames.pkl", "wb"))
