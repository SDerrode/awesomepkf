#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module SeedGenerator
--------------------
Génère et gère des graines aléatoires reproductibles de manière thread-safe,
en utilisant numpy.random.SeedSequence et secrets pour la graine initiale.
"""

from __future__ import annotations
import secrets
import threading
import logging
from typing import Optional
import numpy as np

# ----------------------------------------------------------------------
# Configuration du logging
# ----------------------------------------------------------------------
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class SeedGenerator:
    """
    Gère des graines aléatoires reproductibles et thread-safe.

    Utilise numpy.random.SeedSequence pour dériver des RNG indépendants.
    Permet de créer des sous-générateurs indépendants à partir d'une graine
    principale (maîtresse).
    """

    def __init__(self, seed_key: Optional[int] = None, verbose: int = 0) -> None:
        """
        Parameters
        ----------
        seed_key : Optional[int]
            Graine initiale (si None, une graine forte est générée via secrets).
        verbose : int
            Niveau de verbosité (0, 1 ou 2).
        """
        if __debug__:
            if verbose not in [0, 1, 2]:
                raise ValueError("verbose must be 0, 1 or 2")

        self._lock: threading.Lock = threading.Lock()
        self.verbose: int = verbose

        if seed_key is None:
            seed_key = secrets.randbits(128)
            if __debug__ and self.verbose > 0:
                logger.info(f"[SeedGenerator] Graine forte générée aléatoirement ({seed_key}).")

        self._root_seed: int = seed_key
        self._seed_seq: np.random.SeedSequence = np.random.SeedSequence(self._root_seed)
        self._rng: np.random.Generator = np.random.default_rng(self._seed_seq)

    # ------------------------------------------------------------------
    # Propriétés
    # ------------------------------------------------------------------
    @property
    def rng(self) -> np.random.Generator:
        """Retourne le générateur aléatoire NumPy."""
        return self._rng

    @property
    def seed(self) -> int:
        """Retourne la graine principale utilisée à l’initialisation."""
        return self._root_seed

    # ------------------------------------------------------------------
    # Méthodes principales
    # ------------------------------------------------------------------
    def generate_new_seed(self) -> int:
        """
        Crée et active un nouveau générateur basé sur une sous-séquence indépendante.

        Returns
        -------
        int
            Une nouvelle graine dérivée, utile pour la traçabilité.
        """
        with self._lock:
            new_seq: np.random.SeedSequence = self._seed_seq.spawn(1)[0]
            self._rng = np.random.default_rng(new_seq)
            self._seed_seq = new_seq

            # On récupère un identifiant dérivé pour traçabilité
            derived_seed: int = int(new_seq.entropy)
            if __debug__ and self.verbose > 1:
                logger.info(f"[SeedGenerator] Nouvelle graine dérivée : {derived_seed}")
            return derived_seed

    def __repr__(self) -> str:
        return f"<SeedGenerator seed={self._root_seed} id={id(self):x}>"


# ----------------------------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verbose = 1

    sg1 = SeedGenerator(verbose=verbose)
    print(f"\nsg1 = {sg1}")
    print("Premiers tirages:", sg1.rng.random(3))

    sg1.generate_new_seed()
    print("Après nouvelle graine :", sg1.rng.random(3))

    sg2 = SeedGenerator(42, verbose=verbose)
    print(f"\nsg2 = {sg2}")
    print("Tirages reproductibles :", sg2.rng.random(3))
