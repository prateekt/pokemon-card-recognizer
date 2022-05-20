from typing import List, Sequence, Any

from card_recognizer.run_finding.interval import Interval


def is_sorted(s: Sequence[Any]) -> bool:
    """
    Check if sequence is sorted.

    param s: Sequence to check

    Return:
        True if sequence is sorted
    """
    return all(s[i] <= s[i + 1] for i in range(len(s) - 1))


def find_uninterrupted_runs(series: List[int], query_elem: int) -> List[Interval]:
    """
    Find uninterrupted run intervals of an element in time series.

    param series: The time series of integers
    param query_elem: The element to look for

    Return:
        List of intervals corresponding to uninterrupted runs of element in series
    """
    start = None
    runs: List[Interval] = list()
    for i, curr_elem in enumerate(series):
        if start is None and curr_elem == query_elem:
            start = i
        elif start is not None and curr_elem != query_elem:
            run = Interval(start=start, end=i)
            runs.append(run)
            start = None
    if start is not None:
        run = Interval(start=start, end=len(series))
        runs.append(run)
    return runs


def stitch_with_tol(uninterrupted_runs: List[Interval], tol: int) -> List[Interval]:
    """
    Takes in a list of uninterrupted runs and stitches runs together if they differ only by tol.

    param uninterrupted_runs: List of original runs
    param tol: The tolerance parameter

    Return:
        List of recomputed intervals (w/ stitching)
    """
    if tol < 0:
        raise ValueError("Tol must be >= 0.")
    if len(uninterrupted_runs) < 2:
        return uninterrupted_runs
    new_runs: List[Interval] = list()
    curr_run = uninterrupted_runs[0]
    for next_run in uninterrupted_runs[1:]:
        if next_run.start - curr_run.end <= tol:
            curr_run = Interval(curr_run.start, next_run.end)
        else:
            new_runs.append(curr_run)
            curr_run = next_run
    new_runs.append(curr_run)
    return new_runs


def find_runs_with_tol(series: List[int], query_elem: int, tol: int) -> List[Interval]:
    """
    Find uninterrupted runs of an element in a time series with tolerance to noise where some frames in the detected
    interval may be missing or corrupted:
        1. First finds uninterrupted runs.
        2. Stitches together runs that are close by tol distance
    The tol parameter allows the algorithm to be tolerant to tol (#) frames of noise

    param series: Time series of integers
    query_elem: The element to find runs of in series
    tol: The number of interruption frames allowed in stitching together two uninterrupted runs

    Returns:
        List of detected run intervals of query element
    """
    if tol < 0:
        raise ValueError("Tol must be >= 0.")
    uninterrupted_runs = find_uninterrupted_runs(series=series, query_elem=query_elem)
    return stitch_with_tol(uninterrupted_runs=uninterrupted_runs, tol=tol)
