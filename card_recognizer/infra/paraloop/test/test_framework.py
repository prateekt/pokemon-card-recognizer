import unittest
import card_recognizer.infra.paraloop.paraloop as paraloop


class TestFramework(unittest.TestCase):
    @staticmethod
    def add_1(x: int) -> int:
        return x + 1

    def test_end_to_end(self) -> None:
        """
        Test pool against sequential.
        """
        inputs = (1, 2, 3)
        seq_results = paraloop.loop(
            func=self.add_1, params=inputs, mechanism="sequential"
        )
        pool_results = paraloop.loop(func=self.add_1, params=inputs, mechanism="pool")
        self.assertEqual(seq_results, pool_results)
