import unittest

import numpy as np
import pandas as pd

from data_preprocessing import normalize_features


class TestNormalizeFeaturesFullFit(unittest.TestCase):
    def test_normalize_features_fits_scaler_on_train_val_and_test(self):
        train_df = pd.DataFrame({"sales": [0.0, 10.0]})
        val_df = pd.DataFrame({"sales": [20.0]})
        test_df = pd.DataFrame({"sales": [30.0]})

        train_scaled, val_scaled, test_scaled, scaler, feature_columns = normalize_features(
            train_df,
            val_df,
            test_df,
        )

        self.assertEqual(["sales"], feature_columns)
        self.assertTrue(np.isclose(float(scaler.data_min_[0]), 0.0))
        self.assertTrue(np.isclose(float(scaler.data_max_[0]), 30.0))
        self.assertTrue(np.isclose(float(train_scaled["sales"].iloc[-1]), 1.0 / 3.0))
        self.assertTrue(np.isclose(float(val_scaled["sales"].iloc[0]), 2.0 / 3.0))
        self.assertTrue(np.isclose(float(test_scaled["sales"].iloc[0]), 1.0))


if __name__ == "__main__":
    unittest.main()
