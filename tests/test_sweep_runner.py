"""Smoke tests for fractalsig.runners.sweep_runner."""
from __future__ import annotations

import pytest

from fractalsig.runners.sweep_runner import (
    Cell,
    already_done,
    append_row,
    cells,
    run_one_cell,
)


@pytest.mark.smoke
def test_cells_cartesian():
    out = list(cells(["a"], ["x", "y"], [0, 1]))
    assert len(out) == 4
    assert all(isinstance(c, Cell) for c in out)
    expected = {
        ("a", "x", 0),
        ("a", "x", 1),
        ("a", "y", 0),
        ("a", "y", 1),
    }
    assert {(c.baseline, c.dataset, c.seed) for c in out} == expected


@pytest.mark.smoke
def test_append_writes_header_then_rows(tmp_path):
    p = tmp_path / "r.csv"
    append_row(p, {"baseline": "a", "metric": 1.0})
    append_row(p, {"baseline": "b", "metric": 2.0})
    lines = p.read_text().splitlines()
    assert lines[0] == "baseline,metric"
    assert len(lines) == 3


@pytest.mark.smoke
def test_already_done_resumes_correctly(tmp_path):
    p = tmp_path / "r.csv"
    cell = Cell("a", "x", 0)

    assert not already_done(p, cell)

    append_row(p, {"baseline": "a", "dataset": "x", "seed": 0, "score": 0.5})

    assert already_done(p, cell)
    assert not already_done(p, Cell("a", "x", 1))
    assert not already_done(p, Cell("b", "x", 0))


@pytest.mark.smoke
def test_run_one_cell_raises_until_phase7(tmp_path):
    with pytest.raises(NotImplementedError, match="Phase 7"):
        run_one_cell(Cell("a", "x", 0), tmp_path / "r.csv")
