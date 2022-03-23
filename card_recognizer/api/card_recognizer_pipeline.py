from enum import Enum
from typing import Union, List, Tuple, Optional, Sequence
import collections

from pokemontcgsdk import Card

from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.infra.algo_ops.ops.text import TextOp
from card_recognizer.infra.algo_ops.pipeline.pipeline import Pipeline
from card_recognizer.ocr.pipeline.framework.ffmpeg_op import FFMPEGOp
from card_recognizer.ocr.pipeline.framework.ocr_op import OCRMethod
from card_recognizer.ocr.pipeline.instances import ocr
from card_recognizer.reference.core.build import ReferenceBuild


# CardRecognizerPipeline mode enum
class Mode(Enum):
    SINGLE_IMAGE = 1
    IMAGE_DIR = 2
    VIDEO = 3
    PULLS_IMAGE_DIR = 4
    PULLS_VIDEO = 5


class PullsEstimator(TextOp):
    def _estimate_pulls(
        self, results: Tuple[Sequence[Optional[int]], Sequence[Optional[float]]]
    ) -> List[Card]:
        pred_count = collections.Counter(results[0])
        pred_count.pop(None)
        pulls = [self.set_cards[pred] for pred in pred_count.keys()]
        return pulls

    def __init__(self, set_cards: List[Card]):
        super().__init__(func=self._estimate_pulls)
        self.set_cards = set_cards

    def vis(self) -> None:
        if self.output is not None:
            assert isinstance(self.output, list)
            if len(self.output) == 0:
                print("Pulls Estimator: No Pulls")
            else:
                print("Pulls Estimator:")
                for card in self.output:
                    assert isinstance(card, Card)
                    print(card.name + " (#" + card.number + ")")


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
        set_name: str,
        classification_method: str = "shared_words",
        mode: Mode = Mode.SINGLE_IMAGE,
    ):

        # load classifier
        ref_pkl_path = ReferenceBuild.get_set_pkl_path(set_name=set_name)
        self.classifier = WordClassifier(
            ref_pkl_path=ref_pkl_path,
            vect_method="encapsulation_match",
            classification_method=classification_method,
        )

        # load OCR pipeline
        ocr_pipeline = ocr.basic_ocr_with_text_cleaning_pipeline(
            vocab=self.classifier.reference.vocab, ocr_method=OCRMethod.EASYOCR
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
        elif mode == Mode.PULLS_IMAGE_DIR:
            ops = [
                TextOp(ocr_pipeline.run_on_images),
                TextOp(func=self._classify_func),
                PullsEstimator(set_cards=self.classifier.reference.cards),
            ]
        elif mode == Mode.PULLS_VIDEO:
            ops = [
                FFMPEGOp(),
                TextOp(ocr_pipeline.run_on_images),
                TextOp(func=self._classify_func),
                PullsEstimator(set_cards=self.classifier.reference.cards),
            ]

        else:
            raise ValueError("Unsupported mode: " + str(mode))
        super().__init__(ops=ops)


if __name__ == "__main__":
    pipeline = CardRecognizerPipeline(
        set_name="Vivid Voltage", mode=Mode.PULLS_VIDEO
    )
    #    r = pipeline.exec(inp='/Users/tandonp/Desktop/VID_20220213_132037.mp4')
#    r = pipeline.exec("/Users/tandonp/Desktop/frames_test")
#    LOZ = 1
    r = pipeline.exec(inp="/home/borg1/Desktop/vivid_voltage_test_videos/VID_20220316_214213.mp4")
    print(r)
    LOZ=1
#    r = pipeline.run_on_images("/home/borg1/Desktop/ttframes", mechanism="sequential")
#    pickle.dump(r, open("results_frames.pkl", "wb"))
