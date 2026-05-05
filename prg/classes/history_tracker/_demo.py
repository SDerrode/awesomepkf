"""Demo / smoke test for HistoryTracker.

Run with::

    python -m prg.classes.history_tracker._demo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prg.classes.history_tracker import HistoryTracker


@dataclass
class SimpleStep:
    iter: int
    x: float
    new_x: float
    diff: float


class A:
    """Toy class to illustrate the usage of HistoryTracker."""

    def __init__(self, x0: float = 1.0, verbose: int = 1):
        assert isinstance(x0, (int, float)), "x0 must be a number"
        assert verbose in [0, 1, 2], "verbose must be 0, 1 or 2"

        self.x = float(x0)
        self.verbose = verbose
        self.history = HistoryTracker(verbose=verbose)

    def iterate_gen(self, n: int | None = None):
        k = 0
        while n is None or k < n:
            new_x = np.cos(self.x)
            diff = abs(new_x - self.x)
            step = SimpleStep(iter=k, x=self.x, new_x=new_x, diff=diff)
            self.history.record(step)
            yield step
            self.x = new_x
            k += 1

    def iterate_list(self, n: int):
        """Returns the complete list of iterations."""
        assert isinstance(n, int) and n > 0, "n must be a positive integer"
        return list(self.iterate_gen(n))


if __name__ == "__main__":
    verbose = 1
    graph_dir = Path("data") / "plot"
    tracker_dir = Path("data") / "historyTracker"
    graph_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)

    a = A(x0=1.0, verbose=verbose)
    for step in a.iterate_gen(5):
        print(step)

    a.history.plot(
        title="Evolution of x",
        list_param=["x"],
        list_label=["x"],
        list_covar=[None],
        window={"xmin": 0, "xmax": len(a.history)},
        show=False,
        base_dir=str(graph_dir),
    )
    a.history.save_pickle(str(tracker_dir / "history_run_a.pkl"))
