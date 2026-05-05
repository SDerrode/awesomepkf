"""Tests for the data simulators (linear and nonlinear) and HistoryTracker.plot."""

from __future__ import annotations

import matplotlib as mpl

# Ensure plot tests run headless (no display).
mpl.use("Agg")

import numpy as np
import pytest

from prg.base_classes.simulator_linear import LinearDataSimulator
from prg.base_classes.simulator_nonlinear import NonLinearDataSimulator
from prg.classes.history_tracker import HistoryTracker
from prg.utils.exceptions import ParamError

SEED = 42
N_SHORT = 30


# =====================================================================
# LinearDataSimulator
# =====================================================================


class TestLinearSimulatorInit:

    def test_valid_init(self, tmp_path):
        """Constructor wires up dim/param without raising."""
        sim = LinearDataSimulator(
            model_name="model_x1_y1_AQ_classic",
            N=N_SHORT,
            sKey=SEED,
            data_file_name="dummy.csv",
            verbose=0,
            withoutX=False,
        )
        assert sim.model_name == "model_x1_y1_AQ_classic"
        assert sim.N == N_SHORT
        assert sim.sKey == SEED
        assert sim.param.dim_x == 1
        assert sim.param.dim_y == 1

    def test_default_filename_used_when_none(self):
        """data_file_name=None → ``dataLinear_<model>.csv``."""
        sim = LinearDataSimulator(
            model_name="model_x1_y1_AQ_classic",
            N=N_SHORT,
            sKey=SEED,
            data_file_name=None,
            verbose=0,
            withoutX=False,
        )
        assert sim.data_file_name == "dataLinear_model_x1_y1_AQ_classic.csv"

    def test_invalid_verbose_raises(self):
        with pytest.raises(ParamError, match="verbose"):
            LinearDataSimulator(
                model_name="model_x1_y1_AQ_classic",
                N=N_SHORT,
                sKey=SEED,
                data_file_name="x.csv",
                verbose=99,
                withoutX=False,
            )

    def test_invalid_N_raises(self):
        with pytest.raises(ParamError, match="N must be"):
            LinearDataSimulator(
                model_name="model_x1_y1_AQ_classic",
                N=0,
                sKey=SEED,
                data_file_name="x.csv",
                verbose=0,
                withoutX=False,
            )

    def test_negative_skey_raises(self):
        with pytest.raises(ParamError, match="sKey"):
            LinearDataSimulator(
                model_name="model_x1_y1_AQ_classic",
                N=N_SHORT,
                sKey=-1,
                data_file_name="x.csv",
                verbose=0,
                withoutX=False,
            )


class TestLinearSimulatorRun:

    def test_run_writes_csv(self, tmp_path, monkeypatch):
        """End-to-end: run the simulator and verify the CSV is on disk with rows."""
        monkeypatch.chdir(tmp_path)  # make the relative ``data/datafile/`` land in tmp
        sim = LinearDataSimulator(
            model_name="model_x1_y1_AQ_classic",
            N=N_SHORT,
            sKey=SEED,
            data_file_name="testL.csv",
            verbose=0,
            withoutX=False,
        )
        sim.run()
        out = tmp_path / "data" / "datafile" / "testL.csv"
        assert out.is_file(), f"missing output file: {out}"
        # Header + N+1 rows (initial state + N steps).
        assert sum(1 for _ in out.open()) == N_SHORT + 2

    def test_run_without_x_omits_state_columns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sim = LinearDataSimulator(
            model_name="model_x1_y1_AQ_classic",
            N=10,
            sKey=SEED,
            data_file_name="testL_noX.csv",
            verbose=0,
            withoutX=True,
        )
        sim.run()
        out = tmp_path / "data" / "datafile" / "testL_noX.csv"
        first_line = out.open().readline().strip()
        # withoutX=True → only Y columns.
        assert first_line.startswith("Y"), f"unexpected header: {first_line}"
        assert "X" not in first_line


# =====================================================================
# NonLinearDataSimulator
# =====================================================================


class TestNonLinearSimulator:

    def test_valid_init(self):
        sim = NonLinearDataSimulator(
            model_name="model_x1_y1_pairwise",
            N=N_SHORT,
            sKey=SEED,
            data_file_name="dummy.csv",
            verbose=0,
            withoutX=False,
        )
        assert sim.param.dim_x == 1
        assert sim.param.dim_y == 1
        assert sim.data_file_name == "dummy.csv"

    def test_default_filename_used_when_none(self):
        sim = NonLinearDataSimulator(
            model_name="model_x1_y1_pairwise",
            N=N_SHORT,
            sKey=SEED,
            data_file_name=None,
            verbose=0,
            withoutX=False,
        )
        assert sim.data_file_name == "dataNonLinear_model_x1_y1_pairwise.csv"

    def test_run_writes_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sim = NonLinearDataSimulator(
            model_name="model_x1_y1_pairwise",
            N=N_SHORT,
            sKey=SEED,
            data_file_name="testNL.csv",
            verbose=0,
            withoutX=False,
        )
        sim.run()
        out = tmp_path / "data" / "datafile" / "testNL.csv"
        assert out.is_file()
        assert sum(1 for _ in out.open()) == N_SHORT + 2


# =====================================================================
# HistoryTracker.plot — the previously-untested 6-helper refactor
# =====================================================================


class TestHistoryTrackerPlot:

    @pytest.fixture
    def tracker_with_records(self):
        """Build a HistoryTracker with a few records that mimic a Kalman step."""
        h = HistoryTracker(verbose=0)
        for k in range(20):
            h.record(
                k=k,
                xkp1=np.array([[0.5 * k]]),       # (dim_x=1, 1)
                ykp1=np.array([[0.5 * k + 0.1]]),
                Xkp1_update=np.array([[0.5 * k]]),
                PXXkp1_update=np.array([[0.05]]),
            )
        return h

    def test_plot_observations_only(self, tracker_with_records, tmp_path):
        """Plot a single column without covariance envelope; saves PNG."""
        _fig, axes = tracker_with_records.plot(
            title="Observations",
            list_param=["ykp1"],
            list_label=["y"],
            list_covar=[None],
            window={"xmin": 0, "xmax": 20},
            basename="test_obs",
            show=False,
            base_dir=str(tmp_path),
        )
        assert (tmp_path / "test_obs.png").is_file()
        assert len(axes) == 1

    def test_plot_with_covariance_envelope(self, tracker_with_records, tmp_path):
        """Plot a state estimate together with its ±2σ envelope."""
        _fig, _axes = tracker_with_records.plot(
            title="State estimate",
            list_param=["xkp1", "Xkp1_update"],
            list_label=["x true", "x̂"],
            list_covar=[None, "PXXkp1_update"],
            window={"xmin": 0, "xmax": 20},
            basename="test_state",
            show=False,
            base_dir=str(tmp_path),
        )
        assert (tmp_path / "test_state.png").is_file()

    def test_plot_validates_list_lengths(self, tracker_with_records, tmp_path):
        """Mismatched list_param/list_label/list_covar lengths → ParamError."""
        from prg.utils.exceptions import ParamError

        with pytest.raises(ParamError, match="same length"):
            tracker_with_records.plot(
                title="Bad",
                list_param=["xkp1", "ykp1"],
                list_label=["only one"],          # length mismatch
                list_covar=[None, None],
                window={"xmin": 0, "xmax": 20},
                show=False,
                base_dir=str(tmp_path),
            )

    def test_plot_validates_window_keys(self, tracker_with_records, tmp_path):
        from prg.utils.exceptions import ParamError

        with pytest.raises(ParamError, match="window must contain"):
            tracker_with_records.plot(
                title="Bad window",
                list_param=["ykp1"],
                list_label=["y"],
                list_covar=[None],
                window={"xmin": 0},                # missing xmax
                show=False,
                base_dir=str(tmp_path),
            )

    def test_save_pickle_roundtrip(self, tracker_with_records, tmp_path):
        """Save → load round-trip preserves the recorded history."""
        path = str(tmp_path / "hist.pkl")
        tracker_with_records.save_pickle(path)
        loaded = HistoryTracker.load_pickle(path)
        assert len(loaded) == 20
        assert loaded.last()["k"] == 19
