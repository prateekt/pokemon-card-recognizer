import unittest

from card_recognizer.run_finding.run_finding import (
    find_uninterrupted_runs,
    Range,
    stitch_with_tol,
    find_runs_with_tol,
)


class TestRunFinding(unittest.TestCase):
    def test_uninterrupted_runs(self) -> None:

        # test basic
        series = [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 3]
        runs_1 = find_uninterrupted_runs(series=series, run_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, run_elem=2)
        runs_3 = find_uninterrupted_runs(series=series, run_elem=3)
        self.assertEqual(runs_1, [Range(0, 3), Range(6, 10)])
        self.assertEqual(runs_2, [Range(3, 6)])
        self.assertEqual(runs_3, [Range(10, 11)])

        # test no run
        series = [1, 1, 1]
        runs_1 = find_uninterrupted_runs(series=series, run_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, run_elem=2)
        self.assertEqual(runs_1, [Range(0, 3)])
        self.assertEqual(runs_2, [])

        # test empty series
        series = []
        runs_0 = find_uninterrupted_runs(series=series, run_elem=0)
        self.assertEqual(runs_0, [])

        # test single elem
        series = [1]
        runs_1 = find_uninterrupted_runs(series=series, run_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, run_elem=2)
        self.assertEqual(runs_1, [Range(0, 1)])
        self.assertEqual(runs_2, [])

        # test 2 elem
        series = [1, 2]
        runs_1 = find_uninterrupted_runs(series=series, run_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, run_elem=2)
        self.assertEqual(runs_1, [Range(0, 1)])
        self.assertEqual(runs_2, [Range(1, 2)])

        # test 3 elem
        series = [1, 2, 3]
        runs_1 = find_uninterrupted_runs(series=series, run_elem=1)
        runs_2 = find_uninterrupted_runs(series=series, run_elem=2)
        runs_3 = find_uninterrupted_runs(series=series, run_elem=3)
        self.assertEqual(runs_1, [Range(0, 1)])
        self.assertEqual(runs_2, [Range(1, 2)])
        self.assertEqual(runs_3, [Range(2, 3)])

    def test_stitcher(self) -> None:

        # basic test
        runs = [
            Range(0, 1),
            Range(2, 5),
            Range(10, 12),
            Range(13, 14),
            Range(15, 16),
            Range(100, 101),
        ]
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=4)
        self.assertEqual(stitched, [Range(0, 5), Range(10, 16), Range(100, 101)])
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=5)
        self.assertEqual(stitched, [Range(0, 16), Range(100, 101)])
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=100)
        self.assertEqual(stitched, [Range(0, 101)])
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=0)
        self.assertEqual(stitched, runs)

        # empty runs
        runs = []
        stitched = stitch_with_tol(uninterrupted_runs=runs, tol=0)
        self.assertEqual(stitched, [])

        # test err
        with self.assertRaises(ValueError):
            stitch_with_tol(runs, tol=-1)

    def test_find_runs_with_tol(self) -> None:

        # test basic
        series = [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 3]
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=1, tol=0),
            [Range(0, 3), Range(6, 10)],
        )
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=1, tol=2),
            [Range(0, 3), Range(6, 10)],
        )
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=1, tol=3), [Range(0, 10)]
        )
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=1, tol=4), [Range(0, 10)]
        )
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=2, tol=4), [Range(3, 6)]
        )
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=3, tol=4), [Range(10, 11)]
        )
        self.assertEqual(find_runs_with_tol(series=series, run_elem=4, tol=4), [])

        # test 1 elem
        series = [1]
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=1, tol=1),
            [Range(0, 1)],
        )
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=2, tol=1),
            [],
        )

        # test empty
        series = []
        self.assertEqual(
            find_runs_with_tol(series=series, run_elem=1, tol=1),
            [],
        )
