from typing import List
from collections import namedtuple

Range = namedtuple("Range", ("start", "end"))


def find_uninterrupted_runs(series: List[int], run_elem: int) -> List[Range]:
    """
    Find uninterrupted runs of an element in time series.

    param series: The time series
    param run_elem: The element to look for

    Return:
        List of ranges corresponding to uninterrupted runs of element in series
    """
    start = None
    runs: List[Range] = list()
    for i, curr_elem in enumerate(series):
        if start is None and curr_elem == run_elem:
            start = i
        elif start is not None and curr_elem != run_elem:
            run = Range(start=start, end=i)
            runs.append(run)
            start = None
    if start is not None:
        run = Range(start=start, end=len(series))
        runs.append(run)
    return runs


def stitch_with_tol(uninterrupted_runs: List[Range], tol: int) -> List[Range]:
    """
    Takes in a list of uninterrupted runs and stitches runs together if they differ only by tol.

    param uninterrupted_runs: List of original runs
    param tol: The tolerance

    Return:
        List of new runs (w/ stitching)
    """
    if tol < 0:
        raise ValueError("Tol must be >= 0.")
    if len(uninterrupted_runs) < 2:
        return uninterrupted_runs
    new_runs: List[Range] = list()
    curr_run = uninterrupted_runs[0]
    for next_run in uninterrupted_runs[1:]:
        if next_run.start - curr_run.end <= tol:
            curr_run = Range(curr_run.start, next_run.end)
        else:
            new_runs.append(curr_run)
            curr_run = next_run
    new_runs.append(curr_run)
    return new_runs


def find_runs_with_tol(series: List[int], run_elem: int, tol: int) -> List[Range]:
    """
    Find uninterrupted runs of an element in a time series with tolerance:
        1. First finds uninterrupted runs.
        2. Stitches together runs that are close by tol distance

    Returns:
        Detected runs of element (w/ possible noise)
    """
    if tol < 0:
        raise ValueError("Tol must be >= 0.")
    uninterrupted_runs = find_uninterrupted_runs(series=series, run_elem=run_elem)
    return stitch_with_tol(uninterrupted_runs=uninterrupted_runs, tol=tol)
