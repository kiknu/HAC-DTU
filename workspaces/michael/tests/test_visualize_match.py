import unittest
from pathlib import Path

import numpy as np

from scripts.validation.visualize_match import (
    load_observed,
    metrics_table,
    plot_curve_match,
    plot_residual_heatmap,
)


class VisualizeMatchTests(unittest.TestCase):
    def test_load_observed_simulated_maps_degrees(self):
        observed, degrees = load_observed(
            1, "simulated", "intensity",
            ["0_hor1", "0_top", "0_bottom"], 6,
        )
        self.assertEqual(observed.shape, (60, 3))
        np.testing.assert_allclose(degrees, np.arange(0, 360, 6))

    def test_metrics_table_columns_and_values(self):
        observed = np.array([[1.0, 1.2, 0.9, 0.8], [1.0, 1.1, 1.1, 1.0]])
        predicted = observed + 0.1
        table = metrics_table(predicted, observed, ["a", "b", "c", "d"])
        self.assertEqual(list(table.columns),
                         ["camera", "rmse", "mae", "correlation",
                          "amplitude_ratio"])
        np.testing.assert_allclose(table["rmse"], 0.1)
        self.assertEqual(table["camera"].tolist(), ["a", "b", "c", "d"])

    def test_plots_write_pngs(self, tmpdir=Path("artifacts/forward_match_tests")):
        tmpdir = Path(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        degrees = np.arange(0, 360, 12, dtype=np.float64)
        observed = 1.0 + 0.05 * np.sin(np.radians(degrees))[:, None]
        observed = observed + 0.01 * rng.standard_normal((30, 3))
        predicted = observed + 0.02 * rng.standard_normal((30, 3))
        curves = tmpdir / "curves.png"
        residuals = tmpdir / "residuals.png"
        plot_curve_match(
            predicted, observed, ["a", "b", "c"], degrees, curves,
            title="test", phase_deg=0.0,
        )
        plot_residual_heatmap(
            predicted, observed, ["a", "b", "c"], degrees, residuals,
            title="test", phase_deg=0.0,
        )
        self.assertGreater(curves.stat().st_size, 1_000)
        self.assertGreater(residuals.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
