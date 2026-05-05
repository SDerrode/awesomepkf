"""End-to-end tests for the CLI entry points.

Each test patches ``sys.argv`` and invokes ``prg.run_filter.main`` (or
the per-filter wrapper module's ``main``) the same way the
``awesomepkf-*`` console scripts do. Since the CLI calls ``sys.exit(0)``
on success, we wrap each call in ``pytest.raises(SystemExit)`` and
check the exit code.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from prg import run_filter


@contextmanager
def cli_args(argv: list[str]):
    """Patch sys.argv with the given list (program-name + args)."""
    with patch("sys.argv", argv):
        yield


def _run_main(filter_name: str, argv: list[str]) -> int:
    """Run ``run_filter.main(filter_name)`` with patched argv; return exit code.

    A successful filter run does NOT call ``sys.exit`` — only error paths do.
    """
    with cli_args(argv):
        try:
            run_filter.main(filter_name)
        except SystemExit as e:
            return int(e.code) if e.code is not None else 0
    return 0


# =====================================================================
# Happy path — each filter should run end to end and exit 0
# =====================================================================


class TestCLIHappyPath:

    def test_pkf_simulation(self):
        argv = [
            "awesomepkf-pkf",
            "--linear-model-name", "model_x1_y1_AQ_classic",
            "--N", "30",
            "--s-key", "42",
            "--verbose", "0",
        ]
        assert _run_main("pkf", argv) == 0

    def test_epkf_simulation(self):
        argv = [
            "awesomepkf-epkf",
            "--nonlinear-model-name", "model_x1_y1_pairwise",
            "--N", "30",
            "--s-key", "42",
            "--verbose", "0",
        ]
        assert _run_main("epkf", argv) == 0

    def test_upkf_simulation(self):
        argv = [
            "awesomepkf-upkf",
            "--nonlinear-model-name", "model_x1_y1_pairwise",
            "--N", "30",
            "--s-key", "42",
            "--sigma-set", "wan2000",
            "--verbose", "0",
        ]
        assert _run_main("upkf", argv) == 0

    def test_pf_simulation(self):
        argv = [
            "awesomepkf-pf",
            "--nonlinear-model-name", "model_x1_y1_Sinus_classic",
            "--N", "30",
            "--s-key", "42",
            "--n-particles", "50",
            "--verbose", "0",
        ]
        assert _run_main("pf", argv) == 0

    def test_ppf_simulation(self):
        argv = [
            "awesomepkf-ppf",
            "--nonlinear-model-name", "model_x1_y1_pairwise",
            "--N", "30",
            "--s-key", "42",
            "--n-particles", "50",
            "--verbose", "0",
        ]
        assert _run_main("ppf", argv) == 0


# =====================================================================
# Error paths — each maps to a specific exit code
# =====================================================================


class TestCLIErrors:

    def test_unknown_model_exits_with_param_error(self):
        """Unknown model raises ParamError → exit code 2."""
        argv = [
            "awesomepkf-pkf",
            "--linear-model-name", "model_does_not_exist",
            "--N", "10",
            "--verbose", "0",
        ]
        assert _run_main("pkf", argv) == 2

    def test_pf_rejects_pairwise_exits_with_param_error(self):
        """PF on a pairwise model raises ParamError → exit 2 (PARAMETER ERROR)."""
        argv = [
            "awesomepkf-pf",
            "--nonlinear-model-name", "model_x1_y1_pairwise",
            "--N", "10",
            "--n-particles", "20",
            "--verbose", "0",
        ]
        assert _run_main("pf", argv) == 2

    def test_argparse_conflict_n_with_data_filename(self, capsys):
        """--N and --data-filename together → argparse error → exit 2."""
        argv = [
            "awesomepkf-pkf",
            "--linear-model-name", "model_x1_y1_AQ_classic",
            "--N", "10",
            "--data-filename", "foo.csv",
        ]
        with pytest.raises(SystemExit) as excinfo, cli_args(argv):
            run_filter.main("pkf")
        # argparse exits with 2, not via our _ERROR_TABLE
        assert excinfo.value.code == 2

    def test_argparse_missing_both_n_and_data_filename(self, capsys):
        """Neither --N nor --data-filename → argparse error → exit 2."""
        argv = [
            "awesomepkf-pkf",
            "--linear-model-name", "model_x1_y1_AQ_classic",
        ]
        with pytest.raises(SystemExit) as excinfo, cli_args(argv):
            run_filter.main("pkf")
        assert excinfo.value.code == 2


# =====================================================================
# CLI structure
# =====================================================================


class TestCLIStructure:

    def test_unknown_filter_name_raises_paramerror(self):
        argv = ["whatever", "--linear-model-name", "model_x1_y1_AQ_classic", "--N", "10"]
        # Unknown filter raises ParamError before parse → exit 2 (PARAMETER ERROR).
        assert _run_main("not_a_filter", argv) == 2

    def test_filter_specs_keys_match_main_dispatch(self):
        """Every key in FILTER_SPECS must be a valid filter name for main()."""
        from prg.base_classes.filter_specs import FILTER_SPECS

        # Sanity: the 6 filters we expect.
        assert set(FILTER_SPECS) == {"pkf", "epkf", "upkf", "ukf", "pf", "ppf"}
