import pickle
from enum import Enum
from typing import Union, List, Tuple, Optional

from card_recognizer.classifier.word_classifier import WordClassifier
from card_recognizer.infra.algo_ops.ops.text import TextOp
from card_recognizer.infra.algo_ops.pipeline.pipeline import Pipeline
from card_recognizer.infra.paraloop import paraloop
from card_recognizer.ocr.pipeline.framework.ffmpeg_op import FFMPEGOp
from card_recognizer.ocr.pipeline.framework.ocr_op import OCRMethod
from card_recognizer.ocr.pipeline.framework.ocr_pipeline import OCRPipeline
from card_recognizer.ocr.pipeline.instances import ocr

# CardRecognizerPipeline mode enum
class Mode(Enum):
    SINGLE_IMAGE = 1
    IMAGE_DIR = 2
    VIDEO = 3


class CardRecognizerPipeline(Pipeline):
    def _classify_func(
        self, ocr_words: Union[List[str], List[List[str]]]
    ) -> Union[
        Tuple[Optional[int], Optional[float]],
        Tuple[List[Optional[int]], List[Optional[float]]],
    ]:
        card_pred, probs = self.classifier.classify(
            ocr_words=ocr_words, include_probs=True
        )
        if isinstance(card_pred, list) and isinstance(probs, list):
            card_probs = [
                probs[pred] if pred is not None else None for pred in card_pred
            ]
            return card_pred, card_probs
        elif card_pred is None:
            return None, None
        else:
            return card_pred, probs[card_pred]

    def __init__(
        self,
        ref_pkl_path: str,
        classification_method: str = "shared_words",
        mode: Mode = Mode.SINGLE_IMAGE,
    ):

        # load classifier
        self.classifier = WordClassifier(
            ref_pkl_file=ref_pkl_path,
            vect_method="encapsulation_match",
            classification_method=classification_method,
        )

        # load OCR pipeline
        #        ocr_pipeline = OCRFusion(vocab=self.classifier.vocab)
        ocr_pipeline = ocr.basic_ocr_with_text_cleaning_pipeline(
            vocab=self.classifier.vocab, ocr_method=OCRMethod.EASYOCR
        )

        # make pipeline
        if mode == Mode.VIDEO:
            ops = [
                FFMPEGOp(),
                TextOp(ocr_pipeline.run_on_images),
                TextOp(func=self._classify_func),
            ]
        elif mode == Mode.SINGLE_IMAGE:
            ops = [ocr_pipeline, TextOp(func=self._classify_func)]
        elif mode == Mode.IMAGE_DIR:
            ops = [TextOp(ocr_pipeline.run_on_images), TextOp(func=self._classify_func)]
        else:
            raise ValueError("Unsupported mode: " + str(mode))
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
        files = OCRPipeline.get_image_files(images_dir=images_dir)
        results = paraloop.loop(func=self.exec, params=files, mechanism=mechanism)
        return results


if __name__ == "__main__":
    ref_pkl_file = (
        "/home/borg1/Desktop/pokemon-card-recognizer/card_recognizer/reference/data/ref_build"
        "/brilliant_stars.pkl "
    )
    pipeline = CardRecognizerPipeline(ref_pkl_file, mode=Mode.VIDEO)
#    r = pipeline.exec(inp='/Users/tandonp/Desktop/VID_20220213_132037.mp4')
#    r = pipeline.exec("/Users/tandonp/Desktop/frames_test")
#    r = pipeline.exec(inp="/home/borg1/Desktop/ttframes")
#    r = pipeline.run_on_images("/home/borg1/Desktop/ttframes", mechanism="sequential")
#    pickle.dump(r, open("results_frames.pkl", "wb"))
