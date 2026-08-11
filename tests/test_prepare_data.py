import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.data_paths import IVE_TICKS, PROJECT_ROOT
from utils.prepare_data import clean_mbp1, prepare_kibot_ive


class PrepareDataTests(unittest.TestCase):
    def test_prepare_kibot_aggregates_identical_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "IVE_tickbidask.txt"
            output = root / "clean_IVE_tickbidask.parquet"
            source.write_text(
                "01/02/2024,09:30:00,100.0,99.9,100.1,2\n"
                "01/02/2024,09:30:00,101.0,100.9,101.1,1\n"
                "01/02/2024,09:30:01,102.0,101.9,102.1,3\n",
                encoding="utf-8",
            )

            stats = prepare_kibot_ive(source, output)
            result = pd.read_parquet(output)

            self.assertEqual(stats["input_rows"], 3)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.index.name, "date")
            self.assertAlmostEqual(result.iloc[0]["price"], 100 + 1 / 3)
            self.assertEqual(result.iloc[0]["vol"], 3)
            self.assertEqual(result.iloc[0]["trade_count"], 2)

            second_output = root / "second.parquet"
            prepare_kibot_ive(source, second_output)
            pd.testing.assert_frame_equal(result, pd.read_parquet(second_output))

    def test_paths_are_absolute_and_missing_input_is_actionable(self):
        self.assertTrue(PROJECT_ROOT.is_absolute())
        self.assertTrue(IVE_TICKS.is_relative_to(PROJECT_ROOT))
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.txt"
            with self.assertRaisesRegex(FileNotFoundError, "See DATA.md"):
                prepare_kibot_ive(missing, Path(directory) / "output.parquet")

    def test_clean_mbp1_writes_events_and_reusable_trades(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mbp1.parquet"
            rows = pd.DataFrame(
                {
                    "ts_recv": pd.to_datetime(
                        [
                            "2023-11-01T14:30:00Z",
                            "2023-11-01T14:30:01Z",
                            "2023-11-01T14:30:02Z",
                        ],
                        utc=True,
                    ),
                    "ts_event": pd.to_datetime(
                        [
                            "2023-11-01T14:30:00Z",
                            "2023-11-01T14:30:01Z",
                            "2023-11-01T14:30:02Z",
                        ],
                        utc=True,
                    ),
                    "action": ["A", "T", "T"],
                    "side": ["B", "B", "A"],
                    "price": [5000.0, 5000.25, 5000.0],
                    "size": [1, 2, 3],
                    "flags": [0, 0, 0],
                    "sequence": [1, 2, 3],
                    "bid_px_00": [5000.0, 5000.0, 5000.0],
                    "ask_px_00": [5000.25, 5000.25, 4999.75],
                    "bid_sz_00": [10, 10, 10],
                    "ask_sz_00": [12, 12, 12],
                    "bid_ct_00": [2, 2, 2],
                    "ask_ct_00": [3, 3, 3],
                    "symbol": ["ESZ3", "ESZ3", "ESZ3"],
                }
            )
            rows.to_parquet(source, index=False)

            stats = clean_mbp1(source, root / "processed", symbol="ESZ3")
            events = pd.read_parquet(stats["events"])
            trades = pd.read_parquet(stats["trades"])

            self.assertEqual(stats["invalid_book_rows"], 1)
            self.assertEqual(len(events), 2)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades.loc[0, "trade_sign"], 1)
            self.assertIn("book_imbalance", events.columns)
            self.assertIn("trade_notional", trades.columns)


if __name__ == "__main__":
    unittest.main()
