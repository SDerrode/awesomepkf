#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import secrets
import threading


class SeedManager:
    """
    Gère une graine maîtresse et des sous-générateurs indépendants (reproductibles).
    Conçu pour les environnements multi-thread ou scientifiques.
    """

    _lock = threading.Lock()

    def __init__(self, master_seed=None):
        """
        master_seed : int | None
            Si None → une graine cryptographiquement aléatoire est générée.
        """
        if master_seed is None:
            master_seed = secrets.randbits(128)
        self._master_seed = master_seed
        self._master_seq = np.random.SeedSequence(master_seed)
        self._rng = np.random.default_rng(self._master_seq)
        self._children = {}  # dictionnaire des sous-générateurs nommés

    # ---------------------------------------------------------------------
    # 🧬 Sous-générateurs indépendants
    # ---------------------------------------------------------------------
    def get_subrng(self, name):
        """
        Retourne un sous-générateur reproductible et indépendant.
        Si 'name' existe déjà, retourne toujours le même RNG.
        """
        with self._lock:
            if name not in self._children:
                child_seq = self._master_seq.spawn(1)[0]
                self._children[name] = np.random.default_rng(child_seq)
            return self._children[name]

    # ---------------------------------------------------------------------
    # 🔁 Nouvelle graine maîtresse
    # ---------------------------------------------------------------------
    def reseed(self, new_seed=None):
        """
        Change complètement la graine maîtresse et réinitialise les sous-RNG.
        """
        with self._lock:
            if new_seed is None:
                new_seed = secrets.randbits(128)
            self._master_seed = new_seed
            self._master_seq = np.random.SeedSequence(new_seed)
            self._rng = np.random.default_rng(self._master_seq)
            self._children.clear()

    # ---------------------------------------------------------------------
    # 🧠 Accesseurs
    # ---------------------------------------------------------------------
    @property
    def seed(self):
        return self._master_seed

    @property
    def rng(self):
        return self._rng

    def __repr__(self):
        return f"<SeedManager seed={self._master_seed}, subrngs={len(self._children)}>"


if __name__ == '__main__':
    sm = SeedManager(1234)
    print(sm)

    # Générateur global
    print("Global RNG:", sm.rng.random(3))

    # Deux générateurs indépendants mais reproductibles
    rng_a = sm.get_subrng("thread-A")
    rng_b = sm.get_subrng("thread-B")

    print("A:", rng_a.random(3))
    print("B:", rng_b.random(3))

    # Si tu rappelles avec le même nom → tu retrouves le même RNG
    rng_a2 = sm.get_subrng("thread-A")
    print("A2 (identique à A):", rng_a2.random(3))

    # Nouvelle graine maîtresse
    sm.reseed()
    print("\nAprès reseed:", sm)
    print("Global RNG (nouveau):", sm.rng.random(3))
