"""Iterate (baseline, dataset, seed) cells, run each, append results to CSV.

Phase 7 of the plan wires `run_one_cell` to actual baselines + datasets;
this module's purpose is to define the cell schema, the cartesian iterator,
and the resumable CSV append protocol so downstream code can call them.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cell:
    """One unit of the sweep: train `baseline` on `dataset` with `seed`."""

    baseline: str
    dataset: str
    seed: int


def cells(baselines: list[str], datasets: list[str], seeds: list[int]) -> Iterable[Cell]:
    """Cartesian product of (baseline, dataset, seed)."""
    for b in baselines:
        for d in datasets:
            for s in seeds:
                yield Cell(b, d, s)


def append_row(path: Path, row: dict) -> None:
    """Append `row` to CSV at `path`; write header on first row.

    Args:
        path: Output CSV file. Parent dirs are created if missing.
        row: Dict whose keys define the CSV columns (must be consistent across calls).
    """
    header_needed = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if header_needed:
            w.writeheader()
        w.writerow(row)


def already_done(path: Path, cell: Cell) -> bool:
    """Return True if `path` exists and already contains a row for `cell`.

    Used by the sweep driver to skip cells that have already been run, so that
    a sweep killed mid-flight can be resumed without redoing completed work.
    """
    if not path.exists():
        return False
    with path.open() as f:
        for row in csv.DictReader(f):
            if (
                row.get("baseline") == cell.baseline
                and row.get("dataset") == cell.dataset
                and str(row.get("seed")) == str(cell.seed)
            ):
                return True
    return False


def run_one_cell(cell: Cell, results_csv: Path) -> dict:
    """Train one (baseline, dataset, seed) cell.

    The concrete implementation is wired in Phase 7 once baselines and metrics
    exist. Until then this raises so callers cannot accidentally rely on a
    silent no-op.
    """
    raise NotImplementedError(
        "sweep_runner.run_one_cell is a stub until Phase 7 wires baselines + metrics."
    )
