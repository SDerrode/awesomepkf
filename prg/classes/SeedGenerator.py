"""
Module SeedGenerator
--------------------
Generates and manages reproducible random seeds in a thread-safe manner,
using numpy.random.SeedSequence and secrets for the initial seed.
"""

from __future__ import annotations

import secrets
import threading

import numpy as np

from prg.utils.exceptions import ParamError

__all__ = ["SeedGenerator"]


class SeedGenerator:
    """
    Manages reproducible and thread-safe random seeds.

    Uses numpy.random.SeedSequence to derive independent RNGs.
    Allows creating independent sub-generators from a main (master) seed.
    """

    def __init__(self, seed_key: int | None = None, verbose: int = 0) -> None:
        """
        Parameters
        ----------
        seed_key : int | None
            Initial seed (any integer, positive or negative). If None, a strong seed is generated via secrets.
        verbose : int
            Verbosity level (0, 1 or 2).

        Raises
        ------
        ParamError
            If ``verbose`` does not belong to ``{0, 1, 2}``.
        ParamError
            If ``seed_key`` is provided but is not an integer.
        """
        if __debug__:
            if verbose not in [0, 1, 2]:
                raise ParamError("verbose must be 0, 1 or 2")
            if seed_key is not None and not isinstance(seed_key, int):
                raise ParamError("seed_key must be None or an integer")

        self._lock: threading.Lock = threading.Lock()
        self.verbose: int = verbose

        if seed_key is None:
            seed_key = secrets.randbits(128)

        self._root_seed: int = seed_key
        self._seed_seq: np.random.SeedSequence = np.random.SeedSequence(self._root_seed)
        self._rng: np.random.Generator = np.random.default_rng(self._seed_seq)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def rng(self) -> np.random.Generator:
        """Returns the NumPy random generator."""
        return self._rng

    @property
    def seed(self) -> int:
        """Returns the main seed used at initialisation."""
        return self._root_seed

    # ------------------------------------------------------------------
    # Main methods
    # ------------------------------------------------------------------
    def generate_new_seed(self) -> int:
        """
        Creates and activates a new generator based on an independent sub-sequence.

        Returns
        -------
        int
            A new derived seed, useful for traceability.
        """
        with self._lock:
            new_seq: np.random.SeedSequence = self._seed_seq.spawn(1)[0]
            self._rng = np.random.default_rng(new_seq)
            self._seed_seq = new_seq

            derived_seed: int = int(new_seq.entropy)
            return derived_seed

    def __repr__(self) -> str:
        return f"<SeedGenerator seed={self._root_seed} id={id(self):x}>"


# ----------------------------------------------------------------------
# Usage example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verbose = 1

    sg1 = SeedGenerator(verbose=verbose)
    print(f"\nsg1 = {sg1}")
    print("First draws:", sg1.rng.random(3))

    sg1.generate_new_seed()
    print("After new seed:", sg1.rng.random(3))

    sg2 = SeedGenerator(42, verbose=verbose)
    print(f"\nsg2 = {sg2}")
    print("Reproducible draws:", sg2.rng.random(3))
