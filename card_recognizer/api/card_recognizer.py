import os
from enum import Enum
from typing import Optional, Any

from algo_ops.ops.op import Op
from algo_ops.pipeline.pipeline import Pipeline
from ocr_ops.framework.op.abstract_ocr_op import EasyOCROp
from ocr_ops.framework.op.ffmpeg_op import FFMPEGOp
from ocr_ops.framework.pipeline.ocr_pipeline import OCRMethod
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
        min_run_length: Optional[int] = 5,
        min_run_conf: Optional[float] = 0.1,
        run_tol: Optional[int] = 10,
    ):

        # load classifier
        ref_pkl_path = ReferenceBuild.get_set_pkl_path(set_name=set_name)
        self.classifier = WordClassifier(
            ref_pkl_path=ref_pkl_path,
            vect_method="encapsulation_match",
            classification_method=classification_method,
        )

        # load OCR pipeline
        self.ocr_pipeline = ocr.basic_ocr_with_text_cleaning_pipeline(
            vocab_words=self.classifier.reference.vocab(),
            ocr_method=OCRMethod.EASYOCR,
        )

        # make pipeline
        if mode == Mode.VIDEO:
            ops = [FFMPEGOp(), self.ocr_pipeline, self.classifier]
        elif mode == Mode.SINGLE_IMAGE:
            ops = [self.ocr_pipeline, self.classifier]
        elif mode == Mode.IMAGE_DIR:
            ops = [self.ocr_pipeline, self.classifier]
        elif mode == Mode.PULLS_IMAGE_DIR:
            ops = [
                self.ocr_pipeline,
                self.classifier,
                PullsEstimator(
                    min_run_length=min_run_length,
                    min_run_conf=min_run_conf,
                    run_tol=run_tol,
                    num_cards_to_select=None,
                ),
                PullsSummary(),
            ]
        elif mode == Mode.PULLS_VIDEO:
            ops = [
                FFMPEGOp(),
                self.ocr_pipeline,
                self.classifier,
                PullsEstimator(
                    min_run_length=min_run_length,
                    min_run_conf=min_run_conf,
                    run_tol=run_tol,
                    num_cards_to_select=None,
                    figs_paging=True,
                ),
                PullsSummary(),
            ]
        elif mode == Mode.BOOSTER_PULLS_IMAGE_DIR:
            ops = [
                self.ocr_pipeline,
                self.classifier,
                PullsEstimator(
                    min_run_length=min_run_length,
                    min_run_conf=min_run_conf,
                    run_tol=run_tol,
                ),
                PullsSummary(),
            ]
        elif mode == Mode.BOOSTER_PULLS_VIDEO:
            ops = [
                FFMPEGOp(),
                self.ocr_pipeline,
                self.classifier,
                PullsEstimator(
                    min_run_length=min_run_length,
                    min_run_conf=min_run_conf,
                    run_tol=run_tol,
                ),
                PullsSummary(),
            ]
        else:
            raise ValueError("Unsupported mode: " + str(mode))
        super().__init__(ops=ops)

    def find_op_by_class(self, op_class: Any) -> Optional[Op]:
        for op in self.ops.values():
            if isinstance(op, op_class):
                return op
        return None

    def set_output_path(self, output_path: Optional[str] = None):
        """
        Set output path for results.
        """
        ffmpeg_op = self.find_op_by_class(op_class=FFMPEGOp)
        if ffmpeg_op is not None:
            ffmpeg_op.image_out_path = os.path.join(
                output_path, "uncompressed_video_frames"
            )
        pulls_estimator_op = self.find_op_by_class(op_class=PullsEstimator)
        if pulls_estimator_op is not None:
            pulls_estimator_op.output_fig_path = output_path
        autosave_path = os.path.join(output_path, "ocr_bounding_boxes")
        self.ocr_pipeline.ocr_op.autosave_output_img_path = autosave_path

    def set_summary_file(self, summary_file: str):
        """
        Set summary file.
        """
        pulls_summary_op = self.find_op_by_class(op_class=PullsSummary)
        if pulls_summary_op is None:
            raise ValueError("There is no pulls summary op found in this pipeline.")
        if pulls_summary_op is not None:
            pulls_summary_op.summary_file = summary_file

    def to_pickle(self, out_pkl_path: str) -> None:

        # temporarily remove un-pickleable elements
        easy_ocr_instance = None
        if isinstance(self.ocr_pipeline.ocr_op, EasyOCROp):
            easy_ocr_instance = self.ocr_pipeline.ocr_op.easy_ocr_reader
            self.ocr_pipeline.ocr_op.easy_ocr_reader = None

        # super call to pickle
        super().to_pickle(out_pkl_path=out_pkl_path)

        # restore state
        if isinstance(self.ocr_pipeline.ocr_op, EasyOCROp):
            self.ocr_pipeline.ocr_op.easy_ocr_reader = easy_ocr_instance
