import os
from enum import Enum
from typing import Optional

from algo_ops.ops.text import TextOp
from algo_ops.pipeline.pipeline import Pipeline
from ocr_ops.framework.op.ffmpeg_op import FFMPEGOp
from ocr_ops.framework.op.ocr_op import OCRMethod
from ocr_ops.instances import ocr

from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.pulls_estimator.pulls_estimator import PullsEstimator
from card_recognizer.pulls_estimator.pulls_summary import PullsSummary
from card_recognizer.reference.core.build import ReferenceBuild


# CardRecognizerPipeline mode enum
class Mode(Enum):
    SINGLE_IMAGE = 1
    IMAGE_DIR = 2
    VIDEO = 3
    PULLS_IMAGE_DIR = 4
    PULLS_VIDEO = 5
    BOOSTER_PULLS_IMAGE_DIR = 6
    BOOSTER_PULLS_VIDEO = 7


class CardRecognizer(Pipeline):
    def __init__(
        self,
        set_name: Optional[str] = "master",
        classification_method: str = "shared_words",
        mode: Mode = Mode.SINGLE_IMAGE,
        suppress_plotly_output: bool = True,
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
            vocab_words=self.classifier.reference.vocab(), ocr_method=OCRMethod.EASYOCR
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
                PullsEstimator(
                    suppress_plotly_output=suppress_plotly_output,
                ),
                PullsSummary(),
            ]
        elif mode == Mode.PULLS_VIDEO:
            ops = [
                FFMPEGOp(),
                TextOp(ocr_pipeline.run_on_images),
                self.classifier,
                PullsEstimator(
                    suppress_plotly_output=suppress_plotly_output,
                ),
                PullsSummary(),
            ]
        elif mode == Mode.BOOSTER_PULLS_IMAGE_DIR:
            ops = [
                TextOp(ocr_pipeline.run_on_images),
                self.classifier,
                PullsEstimator(
                    freq_t=0,
                    suppress_plotly_output=suppress_plotly_output,
                ),
                PullsSummary(),
            ]
        elif mode == Mode.BOOSTER_PULLS_VIDEO:
            ops = [
                FFMPEGOp(),
                TextOp(ocr_pipeline.run_on_images),
                self.classifier,
                PullsEstimator(
                    conf_t=0.25,
                    suppress_plotly_output=suppress_plotly_output,
                ),
                PullsSummary(),
            ]
        else:
            raise ValueError("Unsupported mode: " + str(mode))
        super().__init__(ops=ops)

    def set_output_path(self, output_path: Optional[str] = None):
        """
        Set output path for results.
        """
        for op_name in self.ops.keys():
            op = self.ops[op_name]
            if isinstance(op, FFMPEGOp):
                op.out_path = os.path.join(output_path, "uncompressed_video_frames")
            if isinstance(op, PullsEstimator):
                op.output_fig_path = output_path
