"""Tests for utility I/O helpers."""

from __future__ import annotations

import pytest

from prg.utils.io import read_unknown_file


def test_read_unknown_file_rejects_invalid_nrows_detect(tmp_path) -> None:
    data_file = tmp_path / "sample.csv"
    data_file.write_text("Y0\n1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nrows_detect must be an integer >= 1"):
        read_unknown_file(str(data_file), nrows_detect=0)
