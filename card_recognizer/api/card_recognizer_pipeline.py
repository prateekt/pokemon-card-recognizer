from card_recognizer.classifier.word_classifier import WordClassifier
from card_recognizer.infra.algo_ops.pipeline import Pipeline
from card_recognizer.infra.algo_ops.textops import TextOp
from card_recognizer.ocr.pipeline.framework.ocr_fusion import OCRFusion


class CardRecognizerPipeline(Pipeline):
    def __init__(self, ref_pkl_path: str):

        # load classifier
        classifier = WordClassifier(
            ref_pkl_file=ref_pkl_path, vect_method="encapsulation_match"
        )

        # load OCR pipeline
        ocr_pipeline = OCRFusion(vocab=classifier.vocab)

        # make pipeline
        ops = [ocr_pipeline, TextOp(func=classifier.classify)]
        super().__init__(ops=ops)
