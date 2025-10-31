#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import secrets
import threading

class SeedGenerator:
    """
    Classe pour gérer des graines aléatoires reproductibles et thread-safe.
    Utilise numpy.random.SeedSequence pour dériver des RNG indépendants.
    """

    _lock = threading.Lock()  # pour la sécurité en environnement multi-thread

    def __init__(self, sKey=None):
        if sKey is None:
            # Graine maîtresse aléatoire et forte
            sKey = secrets.randbits(128)
        self._master_seed = sKey
        self._master_seq = np.random.SeedSequence(self._master_seed)
        self._rng = np.random.default_rng(self._master_seq)

    def generateNewRandomSeed(self):
        """Crée une nouvelle séquence indépendante mais traçable."""
        with self._lock:
            new_seq = self._master_seq.spawn(1)[0]  # sous-séquence indépendante
            self._rng = np.random.default_rng(new_seq)
            self._master_seq = new_seq  # on met à jour la séquence courante

    @property
    def seed(self):
        """Retourne la graine maîtresse."""
        return self._master_seed

    @property
    def rng(self):
        """Retourne le générateur NumPy."""
        return self._rng

    def __repr__(self):
        return f"<SeedGenerator seed={self._master_seed}>"

if __name__ == '__main__':
    
    sg1 = SeedGenerator()
    print(f"sg1 = {sg1}")
    print("Premiers tirages:", sg1.rng.random(3))

    sg1.generateNewRandomSeed()
    print("Après nouvelle seed:", sg1.rng.random(3))

    sg2 = SeedGenerator(42)
    print(f"sg2 = {sg2}")
    print("Tirages reproductibles:", sg2.rng.random(3))
