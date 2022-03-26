import collections
import os
from enum import Enum
from typing import List, Tuple, Optional, Sequence

import numpy as np
from algo_ops.ops.text import TextOp
from algo_ops.pipeline.pipeline import Pipeline
from pokemontcgsdk import Card

from card_recognizer.classifier.core.word_classifier import WordClassifier
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
    ) -> List[Tuple[Card, int, float]]:
        """
        Estimates pulls from results.

        param: Results structure (Card Number Predictions List, Confidence Score List)

        return:
            List of Pulls (Card Object, # of frames card appears in, Max Confidence Score)
        """
        pred_count = collections.Counter(results[0])
        if None in pred_count:
            pred_count.pop(None)
        pulls = [
            (
                self.set_cards[pred],
                pred_count[pred],
                float(
                    np.max(
                        [
                            results[1][i]
                            for i in range(len(results[0]))
                            if results[0][i] == pred
                        ]
                    )
                ),
            )
            for pred in pred_count.keys()
        ]
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
            ops = [FFMPEGOp(), TextOp(ocr_pipeline.run_on_images), self.classifier]
        elif mode == Mode.SINGLE_IMAGE:
            ops = [ocr_pipeline, self.classifier]
        elif mode == Mode.IMAGE_DIR:
            ops = [TextOp(ocr_pipeline.run_on_images), self.classifier]
        elif mode == Mode.PULLS_IMAGE_DIR:
            ops = [
                TextOp(ocr_pipeline.run_on_images),
                self.classifier,
                PullsEstimator(set_cards=self.classifier.reference.cards),
            ]
        elif mode == Mode.PULLS_VIDEO:
            ops = [
                FFMPEGOp(),
                TextOp(ocr_pipeline.run_on_images),
                self.classifier,
                PullsEstimator(set_cards=self.classifier.reference.cards),
            ]

        else:
            raise ValueError("Unsupported mode: " + str(mode))
        super().__init__(ops=ops)


if __name__ == "__main__":
    pipeline = CardRecognizerPipeline(set_name="Master", mode=Mode.PULLS_VIDEO)
    in_dir = os.sep + os.path.join(
        "home", "borg1", "Desktop", "vivid_voltage_test_videos"
    )
    videos = [os.path.join(in_dir, video) for video in os.listdir(in_dir)]
    for video in videos:
        print(video)
        r = pipeline.exec(inp=video)
        print([a[0].name for a in r if a[1] > 10 and a[2] > 0.25])
