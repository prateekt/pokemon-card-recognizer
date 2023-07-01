import os
from datetime import datetime

import algo_ops.ops.settings as algo_ops_settings
import ezplotly.settings as ezplotly_settings
from algo_ops.dependency import sys_util
from natsort import natsorted

from card_recognizer.api.card_recognizer import CardRecognizer, OperatingMode

if __name__ == "__main__":
    # setup settings
    ezplotly_settings.SUPPRESS_PLOTS = True
    algo_ops_settings.DEBUG_MODE = True
    timestamp = str(datetime.now())

    # init pipeline and paths
    pipeline = CardRecognizer(
        set_name="Paldea Evolved", mode=OperatingMode.BOOSTER_PULLS_VIDEO
    )
    in_dir = os.sep + os.path.join(
        "media",
        "borg1",
        "Borg12TB",
        "card_recognizer_test_sets",
        "paldea_evolved_etb_gamenerdz_6-18-2023",
    )
    out_dir = os.sep + os.path.join("media", "borg1", "Borg12TB", "card_rec_results")
    pipeline.set_summary_file(
        summary_file=os.path.join(in_dir, "pulls_summary_" + timestamp + ".tsv")
    )
    videos = natsorted(
        [
            os.path.join(in_dir, video)
            for video in os.listdir(in_dir)
            if sys_util.is_video_file(video)
        ]
    )

    # loop over videos
    for video in videos:
        print(video)
        results_path = os.path.join(out_dir, os.path.basename(video), timestamp)
        pipeline_pkl_path = os.path.join(results_path, os.path.basename(video) + ".pkl")
        pipeline.set_output_path(output_path=results_path)
        result = pipeline.exec(inp=video)
        pipeline.vis()
        pipeline.to_pickle(out_pkl_path=pipeline_pkl_path)
        print(result)
    """
    in_file = os.sep + os.path.join(
        "media",
        "borg1",
        "Borg12TB",
        "card_recognizer_test_sets",
        "tessa_3_2022",
        "Y2Mate.is - Brilliant Stars Booster Box Opening PART 1-t8NtWA2_26M-1080p-1647284353120.mp4",
    )
    pipeline = CardRecognizer(set_name="Brilliant Stars", mode=OperatingMode.PULLS_VIDEO)
    out_dir = os.sep + os.path.join("media", "borg1", "Borg12TB", "card_rec_results")
    results_path = os.path.join(out_dir, os.path.basename(in_file), timestamp)
    pipeline_pkl_path = os.path.join(results_path, os.path.basename(in_file) + ".pkl")
    pipeline.set_output_path(output_path=results_path)
    result = pipeline.exec(inp=in_file)
    pipeline.vis()
    pipeline.to_pickle(out_pkl_path=pipeline_pkl_path)
    print(result)
    """
