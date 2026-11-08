import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


Budget = runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_budget.py"))["Budget"]


class BudgetTests(unittest.TestCase):
    def test_initialization_is_not_optimization(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True):
            with patch("time.monotonic", return_value=10):
                budget = Budget(directory, 2)
            with patch("time.monotonic", return_value=100):
                budget.begin()
                self.assertFalse(budget.should_stop())
            with patch("time.monotonic", return_value=103):
                budget.checkpoint(4)
                self.assertTrue(budget.should_stop())
                budget.finish()
            timing = json.loads((Path(directory) / "optimization-timing.json").read_text())
            self.assertEqual(timing["initialization_seconds"], 90)
            self.assertEqual(timing["optimization_elapsed_seconds"], 3)
            self.assertEqual(timing["termination_reason"], "time_limit")
            self.assertTrue(timing["graceful"])
            history = (Path(directory) / "optimization-events.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(history[0])["score"], 4)

    def test_invalid_limit_and_stop_marker(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True):
            for limit in (0, -1, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    Budget(directory, limit)
            marker = Path(directory) / "stop"
            marker.touch()
            with patch.dict("os.environ", {"REAXKIT_STOP_FILE": str(marker)}):
                budget = Budget(directory)
                self.assertTrue(budget.should_stop())


if __name__ == "__main__":
    unittest.main()
