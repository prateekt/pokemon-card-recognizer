import os
from typing import Optional, List

import ezplotly as ep
import numpy as np
from ezplotly import EZPlotlyPlot

from card_recognizer.pulls_filter.pull_stats import PullStats


def plot_pull_time_series(
    pull_stats: PullStats,
    outfile: Optional[str] = None,
    suppress_output: bool = True,
) -> None:
    """
    Plot time series of card detections.

    param pull_stats: Precomputed pull statistics
    outfile: Path to output file to save figure
    suppress_output: Whether to suppress output
    """

    # unpack tuple
    frame_card_predictions = pull_stats.frame_card_predictions
    reference = pull_stats.reference

    # make plot
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
    ep.plot_all(h, height=500, outfile=outfile, suppress_output=suppress_output)


def plot_metrics(
    pull_stats: PullStats,
    outfile: Optional[str] = None,
    suppress_output: bool = True,
) -> None:
    """
    Plots metrics such as card detection frequencies, confidence score distributions, and confidence score maximum
    per card.

    param pull_stats: Precomputed pull statistics
    outfile: Path to output file to save figure
    suppress_output: Whether to suppress output
    """

    # unpack tuple
    unique_cards = pull_stats.unique_cards
    reference = pull_stats.reference
    card_frequencies = pull_stats.card_frequencies
    frame_card_predictions = pull_stats.frame_card_predictions
    max_confidence_scores = pull_stats.max_confidence_scores

    # make labels
    unique_card_names = [
        reference.cards[pull].name + " (#" + str(reference.cards[pull].number) + ")"
        for pull in unique_cards
    ]
    unique_card_nums = [
        "#" + str(reference.cards[pull].number) for pull in unique_cards
    ]

    # card detection frequencies
    h: List[Optional[EZPlotlyPlot]] = [None] * (len(unique_card_names) + 2)
    h[0] = ep.bar(
        x=unique_card_nums,
        y=card_frequencies,
        ylabel="Frame Count",
        x_dtick=1,
        title="Card Detection Frequency",
        text=[str(c) for c in card_frequencies],
    )

    # conf score distributions and max
    for i in range(len(unique_card_names)):
        frames = [
            j
            for j, pull in enumerate(frame_card_predictions)
            if pull.card_index_in_reference == unique_cards[i]
        ]
        conf_scores = [frame_card_predictions[f].conf for f in frames]
        h[i + 1] = ep.violin(
            y=conf_scores,
            name=unique_card_nums[i],
            ylabel="Confidence Score",
            title="Detection Confidence Scores Distribution",
        )
    h[-1] = ep.bar(
        x=unique_card_names,
        y=max_confidence_scores,
        ylabel="Max Conf.",
        x_dtick=1,
        title="Max Confidence Score",
        text=[str(round(s, 2)) for s in max_confidence_scores],
        ylim=[0, 1.0],
        y_dtick=0.25,
    )

    # plot
    panels = [1]
    panels.extend([2 for _ in range(len(unique_card_names))])
    panels.extend([3])
    ep.plot_all(
        h, panels=panels, height=600, outfile=outfile, suppress_output=suppress_output
    )


def plot_error_surface(
    pull_stats: PullStats,
    outfile: Optional[str] = None,
    suppress_output: bool = True,
) -> None:
    """
    Plots error surface.

    param pull_stats: Precomputed pull statistics
    outfile: Path to output file to save figure
    suppress_output: Whether to suppress output
    """

    # unpack tuple
    card_frequencies = pull_stats.card_frequencies
    max_confidence_scores = pull_stats.max_confidence_scores

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
    ep.plot_all(h, height=300, outfile=outfile, suppress_output=suppress_output)


def plot_pull_stats(
    pull_stats: PullStats,
    output_fig_path: Optional[str] = None,
    suppress_plotly_output: bool = True,
    prefix: str = "out",
) -> None:
    """
    Make all plots for pull statistics.
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
        pull_stats=pull_stats,
        suppress_output=suppress_plotly_output,
        outfile=time_series_fig_path,
    )
    plot_metrics(
        pull_stats=pull_stats,
        suppress_output=suppress_plotly_output,
        outfile=metrics_fig_path,
    )
    plot_error_surface(
        pull_stats=pull_stats,
        suppress_output=suppress_plotly_output,
        outfile=error_surface_fig_path,
    )
