import os
from pathlib import Path
from typing import Optional, List, Dict

import ezplotly as ep
import ezplotly.settings as plot_settings
import numpy as np
from ezplotly import EZPlotlyPlot

from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    Run,
)
from card_recognizer.reference.core.build import ReferenceBuild


def _dedup(strs: List[str]) -> List[str]:
    """
    De-duplicate time series to avoid labels overlap. Replace duplicates with index ordering.

    param strs: Strings list to dedup

    Return:
        strs: de-duped strs
    """
    encountered: Dict[str, int] = dict()
    for i, s in enumerate(strs):
        if s in encountered.keys():
            strs[i] += " (" + str(encountered[s] + 1) + ")"
            encountered[s] += 1
        else:
            encountered[s] = 1
    return strs


def plot_pull_time_series(
    frame_card_predictions: CardPredictionResult,
    outfile: Optional[str] = None,
) -> None:
    """
    Plot time series of card detections.

    param frame_card_predictions: CardPredictionResult to plot
    outfile: Path to output file to save figure
    """
    reference = ReferenceBuild.get(frame_card_predictions.reference_set)
    y_labels = [None] * frame_card_predictions.num_frames
    for pull in frame_card_predictions:
        y_labels[pull.frame_index] = (
            reference.cards[pull.card_index_in_reference].name
            + " (#"
            + str(reference.cards[pull.card_index_in_reference].number)
            + ")"
        )
    h = ep.scattergl(
        x=list(range(frame_card_predictions.num_frames)),
        y=y_labels,
        xlabel="Frame Number",
        ylabel="Pokemon Card",
        y_dtick=1,
        title="Pokemon Cards Shown in Video",
    )
    ep.plot_all(
        h, height=500, outfile=outfile, suppress_output=plot_settings.SUPPRESS_PLOTS
    )


def plot_metrics(
    runs: List[Run],
    frame_card_predictions: CardPredictionResult,
    outfile: Optional[str],
) -> None:
    """
    Plots metrics such as card detection frequencies, confidence score distributions, and confidence score maximum
    per card.

    param runs: List of card runs to plot
    param frame_card_predictions: CardPredictionResult to plot
    param outfile: Path to output file to save figure
    """

    # unpack
    pulls = [run.card_index for run in runs]
    reference = ReferenceBuild.get(frame_card_predictions.reference_set)
    card_frequencies = [len(run) for run in runs]
    max_confidence_scores = [run.max_confidence_score for run in runs]
    selection_scores = [run.selection_score for run in runs]

    # make labels
    pull_card_names = [
        reference.cards[pull].name
        + " (#"
        + str(reference.cards[pull].number)
        + ") <br> frames "
        + str(runs[i].interval)
        for i, pull in enumerate(pulls)
    ]
    pull_card_runs = _dedup(
        ["#" + str(reference.cards[pull].number) for i, pull in enumerate(pulls)]
    )

    # card detection frequencies
    h: List[Optional[EZPlotlyPlot]] = [None] * (len(runs) + 3)
    h[0] = ep.bar(
        x=pull_card_runs,
        y=card_frequencies,
        ylabel="Frame Count",
        x_dtick=1,
        title="Card Detection Frequency",
        text=[str(c) for c in card_frequencies],
    )

    # conf score distributions and max
    for i in range(len(pull_card_names)):
        frames = [
            j
            for j, pull in enumerate(frame_card_predictions)
            if pull.card_index_in_reference == pulls[i]
        ]
        conf_scores = [frame_card_predictions[f].conf for f in frames]
        h[i + 1] = ep.violin(
            y=conf_scores,
            name=pull_card_runs[i],
            ylabel="Conf. Score",
            title="Detection Confidence Scores Distribution",
        )
    h[-2] = ep.bar(
        x=pull_card_runs,
        y=max_confidence_scores,
        ylabel="Max Conf.",
        x_dtick=1,
        title="Max Confidence Score",
        text=[str(round(s, 2)) for s in max_confidence_scores],
        ylim=[0, 1.0],
        y_dtick=0.25,
    )
    h[-1] = ep.bar(
        x=pull_card_names,
        y=selection_scores,
        ylabel="Sel. Score",
        x_dtick=1,
        title="Selection Score",
        text=[str(round(s, 2)) for s in selection_scores],
    )

    # plot
    panels = [1]
    panels.extend([2 for _ in range(len(pull_card_names))])
    panels.extend([3, 4])
    ep.plot_all(
        h,
        panels=panels,
        height=600,
        outfile=outfile,
        suppress_output=plot_settings.SUPPRESS_PLOTS,
    )


def plot_paged_metrics(
    frame_card_predictions: CardPredictionResult,
    num_runs_per_page: int = 10,
    outfile: Optional[str] = None,
) -> None:
    """
    Plots metrics such as card detection frequencies, confidence score distributions, and confidence score maximum
    per card.

    param frame_card_predictions: CardPredictionResult to plot
    param num_runs_per_page: The number of runs to put on a fig page
    outfile: Path to output file to save figure
    """

    # determine num pages
    all_runs = frame_card_predictions.runs
    num_pages = int(np.ceil(len(all_runs) / num_runs_per_page))

    # make plot pages
    for page_num in range(0, num_pages):

        # determine data to put on page
        start = page_num * num_runs_per_page
        stop = (page_num + 1) * num_runs_per_page
        page_runs = all_runs[start:stop]
        page_out_file = outfile
        if page_out_file is not None:
            base_file = Path(os.path.basename(outfile)).stem
            new_base_file = base_file + "_page" + str(page_num + 1)
            page_out_file = page_out_file.replace(base_file, new_base_file)

        # make page
        plot_metrics(
            runs=page_runs,
            frame_card_predictions=frame_card_predictions,
            outfile=page_out_file,
        )


def plot_error_surface(
    runs: List[Run],
    outfile: Optional[str] = None,
) -> None:
    """
    Plots error surface.

    param runs: List of card runs to plot
    param outfile: Path to output file to save figure
    """

    # unpack tuple
    card_frequencies = [len(run) for run in runs]
    max_confidence_scores = [run.max_confidence_score for run in runs]

    # plot error surface in filter tradeoff
    x_counts_range = list(range(0, np.max(card_frequencies)))
    y_conf_range = np.arange(0.0, 1.05, 0.05)
    kept_num_cards = np.zeros((len(x_counts_range), len(y_conf_range)), dtype=int)
    for i, x_t in enumerate(x_counts_range):
        for j, y_t in enumerate(y_conf_range):
            kept_num_cards[i, j] = np.sum(
                [
                    card_frequencies[i] >= x_t and max_confidence_scores[i] >= y_t
                    for i in range(len(card_frequencies))
                ]
            )
    h = ep.heatmap(
        z=np.abs(kept_num_cards.T - 10),
        xlabels=x_counts_range,
        ylabels=y_conf_range,
        xlabel="Card Detection Frequency Threshold",
        ylabel="Confidence Score Threshold",
        title="Error Surface",
    )
    ep.plot_all(
        h, height=300, outfile=outfile, suppress_output=plot_settings.SUPPRESS_PLOTS
    )


def plot_pull_stats(
    card_prediction_result: CardPredictionResult,
    output_fig_path: Optional[str] = None,
    prefix: str = "out",
    figs_paging: bool = False,
) -> None:
    """
    Make all plots for pull statistics.

    param card_prediction_result: CardPredictionResult to plot
    output_fig_path: Path to where output figs should go
    prefix: Fig prefix for output figs
    figs_paging: Whether figs should be paged.
    """
    if output_fig_path is not None:
        os.makedirs(output_fig_path, exist_ok=True)
        time_series_fig_path = os.path.join(
            output_fig_path, prefix + "_frame_prediction_time_series.png"
        )
        metrics_fig_path = os.path.join(output_fig_path, prefix + "_metrics.png")
        error_surface_fig_path = os.path.join(
            output_fig_path, prefix + "_error_surface.png"
        )
    else:
        time_series_fig_path = None
        metrics_fig_path = None
        error_surface_fig_path = None
    plot_pull_time_series(
        frame_card_predictions=card_prediction_result,
        outfile=time_series_fig_path,
    )
    if figs_paging:
        plot_paged_metrics(
            frame_card_predictions=card_prediction_result,
            outfile=metrics_fig_path,
        )
    else:
        plot_metrics(
            runs=card_prediction_result.runs,
            frame_card_predictions=card_prediction_result,
            outfile=metrics_fig_path,
        )
    plot_error_surface(
        runs=card_prediction_result.runs,
        outfile=error_surface_fig_path,
    )
