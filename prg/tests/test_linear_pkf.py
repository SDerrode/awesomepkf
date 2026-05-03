"""Tests for the Linear Pairwise Kalman Filter (PKF)."""

import numpy as np
import pytest

from prg.classes.Linear_PKF import Linear_PKF
from prg.utils.exceptions import ParamError

SEED = 42
N_SHORT = 100
N_CALIB = 500


class TestLinearPKFInit:

    def test_valid_init(self, param_x1y1):
        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        assert pkf.dim_x == param_x1y1.dim_x
        assert pkf.dim_y == param_x1y1.dim_y

    def test_invalid_param_type(self):
        with pytest.raises(TypeError):
            Linear_PKF("not_a_param")

    def test_invalid_skey(self, param_x1y1):
        with pytest.raises(ParamError):
            Linear_PKF(param_x1y1, sKey=3.14)

    def test_invalid_verbose(self, param_x1y1):
        with pytest.raises(ParamError):
            Linear_PKF(param_x1y1, verbose=5)


class TestLinearPKFOutputShapes:

    def test_output_length(self, param_x1y1):
        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        results = pkf.process_N_data(N=N_SHORT)
        # process_N_data(N) yields N+1 steps: initial estimate (k=0) + N updates
        assert len(results) == N_SHORT + 1

    def test_output_tuple_shapes_x1y1(self, param_x1y1):
        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        results = pkf.process_N_data(N=10)
        for k, x_true, y_obs, x_pred, x_upd in results:
            assert isinstance(k, int)
            assert x_true.shape == (param_x1y1.dim_x, 1)
            assert y_obs.shape  == (param_x1y1.dim_y, 1)
            assert x_pred.shape == (param_x1y1.dim_x, 1)
            assert x_upd.shape  == (param_x1y1.dim_x, 1)

    def test_output_tuple_shapes_x2y2(self, param_x2y2):
        pkf = Linear_PKF(param_x2y2, sKey=SEED)
        results = pkf.process_N_data(N=10)
        for k, x_true, y_obs, x_pred, x_upd in results:
            assert x_true.shape == (param_x2y2.dim_x, 1)
            assert y_obs.shape  == (param_x2y2.dim_y, 1)
            assert x_pred.shape == (param_x2y2.dim_x, 1)
            assert x_upd.shape  == (param_x2y2.dim_x, 1)

    def test_step_indices_are_sequential(self, param_x1y1):
        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        results = pkf.process_N_data(N=N_SHORT)
        ks = [r[0] for r in results]
        assert ks == list(range(N_SHORT + 1))

    def test_invalid_N(self, param_x1y1):
        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        with pytest.raises(ParamError):
            pkf.process_N_data(N=0)
        with pytest.raises(ParamError):
            pkf.process_N_data(N=-1)


class TestLinearPKFCalibration:
    """Calibration test: checks the filter tracks the true state reasonably well."""

    def _compute_mean_nees(self, results, dim_x):
        """Compute NEES from (x_true - x_update) using the history dataframe."""
        errors = []
        for k, x_true, y_obs, x_pred, x_upd in results:
            if x_true is not None:
                err = float(np.sum((x_true - x_upd) ** 2))
                errors.append(err)
        return np.mean(errors) if errors else None

    def test_filter_tracks_x1y1(self, param_x1y1):
        """Updated estimate should be closer to truth than a naive zero estimator."""
        pkf = Linear_PKF(param_x1y1, sKey=SEED)
        results = pkf.process_N_data(N=N_CALIB)
        mse_filter = np.mean([(x_true - x_upd)**2 for _, x_true, _, _, x_upd in results if x_true is not None])
        mse_naive  = np.mean([x_true**2 for _, x_true, _, _, _ in results if x_true is not None])
        assert mse_filter < mse_naive, f"Filter MSE ({mse_filter:.3f}) not better than zero estimator ({mse_naive:.3f})"

    def test_filter_tracks_x2y2(self, param_x2y2):
        pkf = Linear_PKF(param_x2y2, sKey=SEED)
        results = pkf.process_N_data(N=N_CALIB)
        mse_filter = np.mean([np.sum((x_true - x_upd)**2) for _, x_true, _, _, x_upd in results if x_true is not None])
        mse_naive  = np.mean([np.sum(x_true**2) for _, x_true, _, _, _ in results if x_true is not None])
        assert mse_filter < mse_naive, f"Filter MSE ({mse_filter:.3f}) not better than zero estimator ({mse_naive:.3f})"
