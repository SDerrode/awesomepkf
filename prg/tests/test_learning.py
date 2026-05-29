"""Tests for the 1D PMM method-of-moments estimator."""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from prg import run_learn_pmm
from prg.learning.pmm_moments import (
    PMMParams,
    estimate_pmm_params,
    pmm_to_linear_params,
    validate_pmm,
)
from prg.utils.exceptions import NumericalError, ParamError

SEED = 42
N_LONG = 50_000
TOL = 0.03

# Reference PMM parameters — same pattern as Generator_Param_PMM(paramsetnumber=0)
# from the original KalmanApp code: HMM-compatible on c and d, but e deviates,
# which guarantees ``is_hmm()`` returns False while keeping Γ comfortably PD.
A_REF = 0.9
B_REF = -0.2
C_REF = A_REF * B_REF**2     # = a*b² (HMM compat for c)
D_REF = A_REF * B_REF        # = a*b  (HMM compat for d)
E_REF = A_REF * B_REF - 0.4  # off the HMM submanifold (e ≠ a*b)
REF_PARAMS = PMMParams(A_REF, B_REF, C_REF, D_REF, E_REF)


def _simulate_pmm(params: PMMParams, n_steps: int, seed: int) -> np.ndarray:
    """Generate a length-``n_steps`` trajectory of the pairwise state (X, Y)."""
    kw = pmm_to_linear_params(params)
    A, B, Pz0 = kw["A"], kw["B"], kw["Pz0"]
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(2), Pz0).reshape(2, 1)
    traj = np.empty((n_steps, 2))
    traj[0] = z.ravel()
    for k in range(1, n_steps):
        w = rng.standard_normal((2, 1))
        z = A @ z + B @ w
        traj[k] = z.ravel()
    return traj


def _assert_params_close(est: PMMParams, ref: PMMParams, tol: float = TOL) -> None:
    for name, true_val, est_val in zip("abcde", ref, est, strict=True):
        assert abs(true_val - est_val) < tol, (
            f"{name}: expected {true_val:.4f}, got {est_val:.4f}"
        )


class TestPMMParams:

    def test_unpacking(self):
        a, b, c, d, e = REF_PARAMS
        assert (a, b, c, d, e) == (A_REF, B_REF, C_REF, D_REF, E_REF)

    def test_as_hmm_zeroes_the_offsets(self):
        hmm = REF_PARAMS.as_hmm()
        assert hmm.is_hmm()
        assert hmm.a == REF_PARAMS.a
        assert hmm.b == REF_PARAMS.b

    def test_ref_is_not_hmm(self):
        assert not REF_PARAMS.is_hmm()


class TestValidation:

    def test_reference_params_are_valid(self):
        assert validate_pmm(REF_PARAMS)

    def test_b_equal_one_rejected(self):
        assert not validate_pmm(PMMParams(0.5, 1.0, 0.1, 0.1, 0.1))
        assert not validate_pmm(PMMParams(0.5, -1.0, 0.1, 0.1, 0.1))

    def test_c_equal_one_rejected(self):
        assert not validate_pmm(PMMParams(0.5, 0.2, 1.0, 0.1, 0.1))

    def test_hmm_projection_is_valid(self):
        params = REF_PARAMS.as_hmm()
        assert validate_pmm(params)


class TestEstimator:

    def test_recovers_synthetic_pmm(self):
        traj = _simulate_pmm(REF_PARAMS, n_steps=N_LONG, seed=SEED)
        est = estimate_pmm_params(traj, x_col=0, y_col=1, standardise=True)
        _assert_params_close(est, REF_PARAMS)

    def test_dataframe_with_string_columns(self):
        traj = _simulate_pmm(REF_PARAMS, n_steps=N_LONG, seed=SEED)
        df = pd.DataFrame(traj, columns=["state", "obs"])
        est = estimate_pmm_params(df, x_col="state", y_col="obs", standardise=True)
        _assert_params_close(est, REF_PARAMS)

    def test_too_few_samples(self):
        with pytest.raises(ParamError):
            estimate_pmm_params(np.zeros((2, 2)))

    def test_wrong_array_shape(self):
        with pytest.raises(ParamError):
            estimate_pmm_params(np.zeros((100, 1)))

    def test_string_col_on_ndarray(self):
        with pytest.raises(ParamError):
            estimate_pmm_params(np.zeros((100, 2)), x_col="foo", y_col="bar")


class TestPmmToLinearParams:

    def test_returns_expected_shapes(self):
        kw = pmm_to_linear_params(REF_PARAMS)
        assert kw["A"].shape == (2, 2)
        assert kw["B"].shape == (2, 2)
        assert kw["mQ"].shape == (2, 2)
        assert kw["mz0"].shape == (2, 1)
        assert kw["Pz0"].shape == (2, 2)

    def test_B_squared_recovers_mQ(self):
        kw = pmm_to_linear_params(REF_PARAMS)
        # sqrtm gives the symmetric square root: B @ B == B @ B.T == mQ.
        np.testing.assert_allclose(kw["B"] @ kw["B"].T, kw["mQ"], atol=1e-10)

    def test_B_is_symmetric_real(self):
        kw = pmm_to_linear_params(REF_PARAMS)
        np.testing.assert_allclose(kw["B"], kw["B"].T, atol=1e-10)
        assert kw["B"].dtype == np.float64

    def test_Pz0_is_initial_pairwise_covariance(self):
        kw = pmm_to_linear_params(REF_PARAMS)
        expected = np.array([[1.0, B_REF], [B_REF, 1.0]])
        np.testing.assert_allclose(kw["Pz0"], expected, atol=1e-12)

    def test_invalid_params_raise(self):
        with pytest.raises(NumericalError):
            pmm_to_linear_params(PMMParams(0.5, 1.0, 0.1, 0.1, 0.1))


class TestSampleCsv:
    """Smoke test on the embedded WindFarms sample (real data)."""

    @pytest.fixture(scope="class")
    def sample_path(self):
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "data" / "samples" / "windfarms" / "site1_202210_Month_586_norm.csv"
        if not path.exists():
            pytest.skip(f"sample file missing: {path}")
        return path

    def test_estimator_yields_valid_pmm(self, sample_path):
        df = pd.read_csv(sample_path, index_col=0)
        params = estimate_pmm_params(df, x_col="ActivePower_KWh", y_col="WindSpeed")
        assert validate_pmm(params), f"sample data did not yield a valid PMM: {params}"


# =====================================================================
# CLI — invoked via patched sys.argv so the test exercises argument
# parsing, file reading, validation, and the .npz writer.
# =====================================================================


@contextmanager
def _cli_args(argv: list[str]):
    with patch("sys.argv", argv):
        yield


class TestFitPkfCLI:

    @pytest.fixture(scope="class")
    def sample_path(self):
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "data" / "samples" / "windfarms" / "site1_202210_Month_586_norm.csv"
        if not path.exists():
            pytest.skip(f"sample file missing: {path}")
        return path

    def test_happy_path_writes_npz(self, sample_path, tmp_path):
        out = tmp_path / "params.npz"
        argv = [
            "awesomepkf-fit-pkf",
            "--data-filename", str(sample_path),
            "--x-col", "ActivePower_KWh",
            "--y-col", "WindSpeed",
            "--output", str(out),
            "--verbose", "0",
        ]
        with _cli_args(argv):
            run_learn_pmm.main()
        assert out.exists()
        # Roundtrip with allow_pickle=False — the file must contain only
        # primitive scalars and float arrays.
        data = np.load(out, allow_pickle=False)
        for key in ("a", "b", "c", "d", "e", "columns", "data"):
            assert key in data.files, f"missing key {key!r} in npz"
        assert list(map(str, data["columns"])) == ["ActivePower_KWh", "WindSpeed"]
        assert data["data"].dtype == np.float64
        assert data["data"].shape == (586, 2)

    def test_missing_file_exits_with_param_error(self, tmp_path):
        argv = [
            "awesomepkf-fit-pkf",
            "--data-filename", str(tmp_path / "does-not-exist.csv"),
            "--verbose", "0",
        ]
        with _cli_args(argv), pytest.raises(SystemExit) as excinfo:
            run_learn_pmm.main()
        assert excinfo.value.code == 2

    def test_entry_point_installed(self):
        """The console script is registered next to the interpreter and resolves to our main()."""
        exe = Path(sys.executable).parent / "awesomepkf-fit-pkf"
        if not exe.exists():
            pytest.skip(f"console script not installed at {exe} (run pip install -e .)")
        result = subprocess.run(
            [str(exe), "--help"],
            check=True, capture_output=True, text=True,
        )
        assert "Estimate 1D linear PMM parameters" in result.stdout
