import unittest

import numpy as np

from cvphr.test.baseline_pose_metrics import evaluate_pose_predictions


class BaselinePoseMetricsTest(unittest.TestCase):
    def test_city_equal_mle_and_thresholds_match_baseline_definition(self):
        positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        targets = np.zeros((3, 2), dtype=np.float32)
        directions = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        direction_targets = np.asarray([[1.0, 0.0]] * 3)
        paths = ["sat_254k_aa_x", "sat_254k_bb_x", "sat_254k_bb_x"]
        scales = {
            "aa": (10.0, 10.0, 1.0, 1.0),
            "bb": (20.0, 20.0, 1.0, 1.0),
        }

        result = evaluate_pose_predictions(
            positions,
            targets,
            directions,
            direction_targets,
            paths,
            city_scales=scales,
        )

        self.assertAlmostEqual(result.metrics["mae_dis"], 10.0)
        self.assertAlmostEqual(result.metrics["mean_distance_error"], 40.0 / 3.0)
        self.assertAlmostEqual(result.metrics["lsr_at_15"], 100.0 / 3.0)
        self.assertAlmostEqual(result.metrics["hsr_at_15"], 100.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
