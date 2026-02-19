import unittest
from datetime import datetime, timedelta, timezone

from analyzer import compute_training_load_and_zones


def _timestamps(seconds):
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    return [start + timedelta(seconds=i) for i in range(seconds)]


class AnalyzerMetricsTests(unittest.TestCase):
    def test_zone_minutes_from_stream(self):
        # 5m in Zone 2 (~68% max), 2m in Zone 4 (~87% max)
        hr_stream = ([130] * 300) + ([165] * 120)
        metrics = compute_training_load_and_zones(
            hr_stream,
            max_hr=190,
            timestamps=_timestamps(len(hr_stream)),
        )

        self.assertAlmostEqual(metrics["zone1_mins"], 0.0, places=2)
        self.assertAlmostEqual(metrics["zone2_mins"], 5.0, places=2)
        self.assertAlmostEqual(metrics["zone3_mins"], 0.0, places=2)
        self.assertAlmostEqual(metrics["zone4_mins"], 2.0, places=2)
        self.assertAlmostEqual(metrics["zone5_mins"], 0.0, places=2)
        self.assertAlmostEqual(metrics["zone_total_mins"], 7.0, places=2)
        self.assertGreater(metrics["load_score"], 0.0)

    def test_load_scales_with_intensity(self):
        easy = compute_training_load_and_zones(
            [115] * 420,  # ~61% max
            max_hr=190,
            timestamps=_timestamps(420),
        )
        hard = compute_training_load_and_zones(
            [170] * 420,  # ~89% max
            max_hr=190,
            timestamps=_timestamps(420),
        )

        self.assertGreater(hard["load_score"], easy["load_score"])


if __name__ == "__main__":
    unittest.main()
