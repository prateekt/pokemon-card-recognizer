from algo_ops.ops.op import Op

from card_recognizer.ocr.dependency.ffmpeg import FFMPEG


class FFMPEGOp(Op):
    """
    Turn the use of FFMPEG video -> frames conversion into an Op that can placed into an OCR pipeline.
    """

    @staticmethod
    def _convert_to_images_wrapper(video_path: str):
        """
        Wrapper function to convert a video into image frames.

        param video_path: Path to video file

        Return:
            images_frame_path: Path to directory containing frame images extracted from video using FFMPEG.

        """
        success, image_frames_path = FFMPEG.convert_video_to_frames(
            video_path=video_path
        )
        if not success:
            raise SystemError("FFMPEG conversion failed on " + str(video_path))
        return image_frames_path

    def __init__(self):
        super().__init__(func=self._convert_to_images_wrapper)

    def vis(self) -> None:
        print("Converted " + str(self.input) + ".")

    def vis_input(self) -> None:
        pass

    def save_input(self, out_path: str) -> None:
        pass

    def save_output(self, out_path) -> None:
        pass
