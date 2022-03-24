import os
import uuid
from typing import Optional

from card_recognizer.infra.api.sys import run_os_command, check_if_installed


class FFMPEG:
    """
    Wrapper for FFMPEG command line utility.
    """

    @staticmethod
    def is_installed() -> bool:
        return check_if_installed("ffmpeg")

    @staticmethod
    def convert_video_to_frames(
        video_path: str,
        out_path: Optional[str] = None,
        fps: int = 10,
        fmt: str = "out%04d.png",
        squelch_output: bool = True,
    ) -> [bool, str]:
        """
        Converts video to frames. Returns true if conversion succeeded.

        param video_path: Path to input video file (*.mp4)
        param out_path: Path to output
        param fps: Frames per second
        param fmt: Format string
        param squelch_output: Whether to squelch output or not

        return:
            True if conversion succeeded
            out_path: Path to video
        """
        if out_path is None:
            out_path = str(uuid.uuid4())
        os.makedirs(out_path, exist_ok=True)
        out_sig = os.path.join(out_path, fmt)
        _, ret_code = run_os_command(
            ["ffmpeg", "-i", video_path, "-vf", "fps=" + str(fps), out_sig],
            squelch_output=squelch_output,
        )
        return ret_code == 0, out_path
