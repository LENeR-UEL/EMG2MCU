import unittest
from pathlib import Path

import numpy as np

from EMG2MCU._paths import output_context, output_root
from EMG2MCU.features import extract_td_features


class PackageTests(unittest.TestCase):
    def test_output_context_restores_default_directory(self):
        default_root = output_root()

        with output_context("artifacts"):
            self.assertEqual(output_root(), Path("artifacts"))

        self.assertEqual(output_root(), default_root)

    def test_time_domain_iemg_feature(self):
        windows = np.array([[[1.0, -2.0, 3.0]]])
        labels = np.array([1])

        features, feature_labels = extract_td_features(
            windows,
            labels,
            features=["IEMG", "RMS"],
        )

        self.assertEqual(features.loc[0, "IEMG_CH1"], 6.0)
        self.assertAlmostEqual(features.loc[0, "RMS_CH1"], np.sqrt(14 / 3))
        np.testing.assert_array_equal(feature_labels, labels)


if __name__ == "__main__":
    unittest.main()
