import os

from natsort import natsorted
from ocr_ops.dependency import sys_util

from card_recognizer.api.card_recognizer import CardRecognizer, Mode

if __name__ == "__main__":
    pipeline = CardRecognizer(set_name="Brilliant Stars", mode=Mode.BOOSTER_PULLS_VIDEO)
    in_dir = os.sep + os.path.join(
        "media",
        "borg1",
        "Borg12TB",
        "card_recognizer_test_sets",
        "brilliant_stars_booster_box_5_2022",
    )
    out_dir = os.sep + os.path.join("media", "borg1", "Borg12TB", "card_rec_results")
    pipeline.set_summary_file(summary_file=os.path.join(in_dir, "pulls_summary.tsv"))
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
            if sys_util.is_video_file(video)
        ]
    )
    for video in videos:
        print(video)
        results_path = os.path.join(out_dir, os.path.basename(video))
        pipeline_pkl_path = os.path.join(results_path, os.path.basename(video) + ".pkl")
        pipeline.set_output_path(output_path=results_path)
        result = pipeline.exec(inp=video)
        pipeline.vis()
        pipeline.to_pickle(out_pkl_path=pipeline_pkl_path)
        print(result)
