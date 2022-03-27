import os
from enum import Enum
from typing import Optional

from algo_ops.ops.text import TextOp
from algo_ops.pipeline.pipeline import Pipeline
from natsort import natsorted

from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.infra.api import sys
from card_recognizer.ocr.pipeline.framework.ffmpeg_op import FFMPEGOp
from card_recognizer.ocr.pipeline.framework.ocr_op import OCRMethod
from card_recognizer.ocr.pipeline.instances import ocr
from card_recognizer.pulls_filter.pulls_filter import PullsFilter
from card_recognizer.pulls_filter.pulls_summary import PullsSummary
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


class CardRecognizerPipeline(Pipeline):
    def __init__(
        self,
        set_name: str,
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
                PullsFilter(
                    suppress_plotly_output=suppress_plotly_output,
                ),
            ]
        elif mode == Mode.PULLS_VIDEO:
            ops = [
                FFMPEGOp(),
                TextOp(ocr_pipeline.run_on_images),
                self.classifier,
                PullsFilter(
                    suppress_plotly_output=suppress_plotly_output,
                ),
            ]
        elif mode == Mode.BOOSTER_PULLS_IMAGE_DIR:
            ops = [
                TextOp(ocr_pipeline.run_on_images),
                self.classifier,
                PullsFilter(
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
                PullsFilter(
                    suppress_plotly_output=suppress_plotly_output,
                ),
                PullsSummary(),
            ]
        else:
            raise ValueError("Unsupported mode: " + str(mode))
        super().__init__(ops=ops)

    def set_output_figs_path(self, output_figs_path: Optional[str] = None):
        for op_name in self.ops.keys():
            op = self.ops[op_name]
            if isinstance(op, PullsFilter):
                op.output_fig_path = output_figs_path


if __name__ == "__main__":
    pipeline = CardRecognizerPipeline(set_name="Master", mode=Mode.BOOSTER_PULLS_VIDEO)
    in_dir = os.sep + os.path.join(
        "home", "borg1", "Desktop", "vivid_voltage_test_videos"
    )
    """
    in_file = os.sep + os.path.join("home", "borg1", "Desktop",
     "Y2Mate.is - Brilliant Stars Booster Box Opening PART 1-t8NtWA2_26M-1080p-1647284353120.mp4")
    r = pipeline.exec(inp=in_file)
    with open('tt_pkl.pkl', 'wb') as fout:
        pickle.dump(r, fout)
    """
    videos = natsorted(
        [
            os.path.join(in_dir, video)
            for video in os.listdir(in_dir)
            if sys.is_video_file(video)
        ]
    )
    for video in videos:
        print(video)
        pipeline.set_output_figs_path(output_figs_path=os.path.basename(video))
        result = pipeline.exec(inp=video)
        pipeline.vis()
        print(result)
