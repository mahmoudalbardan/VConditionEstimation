import unittest

import numpy as np
import pandas as pd
from src.scripts.build_evaluate_model import compute_features_one_cycle, transform_data


class Test(unittest.TestCase):
    def test_compute_features_one_cycle(self):
        cycle_data_fs1 = np.array([1, 2, 3, 4, 5])
        cycle_data_ps2 = np.array([5, 4, 3, 2, 1])

        expected_features = {
            'FS1_mean': 3.0,
            'FS1_max': 5,
            'FS1_min': 1,
            'FS1_25th': 2.0,
            'FS1_50th': 3.0,
            'FS1_75th': 4.0,
            'PS2_mean': 3.0,
            'PS2_max': 5,
            'PS2_min': 1,
            'PS2_25th': 2.0,
            'PS2_50th': 3.0,
            'PS2_75th': 4.0
        }

        features = compute_features_one_cycle(cycle_data_fs1, cycle_data_ps2)
        for key, value in expected_features.items():
            self.assertAlmostEqual(features[key], value, places=2)

    def test_transform_data(self):
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })

        transformed_data, features_to_log = transform_data(data)
        for feature in transformed_data[features_to_log]:
            self.assertTrue((transformed_data[feature] > 0).all())


if __name__ == '__main__':
    unittest.main()
