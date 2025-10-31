#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class ActiveView:
    """
    Vue sur une sous-matrice qui déclenche une fonction callback
    à chaque modification.
    """
    def __init__(self, parent_matrix, rows, cols, callback):
        self._parent = parent_matrix
        self._rows = rows
        self._cols = cols
        self._callback = callback

    def __getitem__(self, key):
        return self._parent[self._rows, self._cols][key]

    def __setitem__(self, key, value):
        self._parent[self._rows, self._cols][key] = value
        self._callback()

    def __repr__(self):
        return repr(self._parent[self._rows, self._cols])


class ParamPKF:
    """
    Classe pour stocker les paramètres d'un filtre de Kalman prédictif.
    """

    def __init__(self, dim_y, dim_x, A, mQ, verbose=0):
        # Vérifications des dimensions
        if not isinstance(dim_y, int) or dim_y <= 0:
            raise ValueError("dim_y doit être un entier > 0")
        if not isinstance(dim_x, int) or dim_x <= 0:
            raise ValueError("dim_x doit être un entier > 0")
        self.dim_y = dim_y
        self.dim_x = dim_x
        self._n = dim_x + dim_y

        # Vérification des matrices A et mQ
        self._A = np.array(A)
        if self._A.shape != (self._n, self._n):
            raise ValueError(f"A doit être carrée de dimension ({self._n},{self._n})")
        self._update_A_views()

        self._mQ = np.array(mQ)
        if self._mQ.shape != (self._n, self._n):
            raise ValueError(f"mQ doit être carrée de dimension ({self._n},{self._n})")
        self._update_Q_views()

        # Verbose
        if verbose not in [0,1,2]:
            raise ValueError("verbose doit être 0, 1 ou 2")
        self.verbose = verbose

        # Vérification initiale
        self._check_consistency()

    # ------------------------------------------------------------------
    # Propriétés et setters pour A
    # ------------------------------------------------------------------
    @property
    def A(self): return self._A

    @A.setter
    def A(self, new_A):
        new_A = np.array(new_A)
        if new_A.shape != (self._n, self._n):
            raise ValueError(f"A doit être carrée de dimension ({self._n},{self._n})")
        self._A = new_A
        self._update_A_views()
        if self.verbose >= 1:
            print("[ParamPKF] A mis à jour")
        self._check_consistency()

    # ------------------------------------------------------------------
    # Propriétés et setters pour mQ
    # ------------------------------------------------------------------
    @property
    def mQ(self): return self._mQ

    @mQ.setter
    def mQ(self, new_Q):
        new_Q = np.array(new_Q)
        if new_Q.shape != (self._n, self._n):
            raise ValueError(f"mQ doit être carrée de dimension ({self._n},{self._n})")
        self._mQ = new_Q
        self._update_Q_views()
        if self.verbose >= 1:
            print("[ParamPKF] mQ mis à jour")
        self._check_consistency()

    # ------------------------------------------------------------------
    # Propriété pour n
    # ------------------------------------------------------------------
    @property
    def n(self): return self._n

    # ------------------------------------------------------------------
    # Getters pour les vues sur A
    # ------------------------------------------------------------------
    @property
    def A_xx(self): return self._A_xx
    @property
    def A_xy(self): return self._A_xy
    @property
    def A_yx(self): return self._A_yx
    @property
    def A_yy(self): return self._A_yy

    # ------------------------------------------------------------------
    # Getters pour les vues sur Q
    # ------------------------------------------------------------------
    @property
    def Q_xx(self): return self._Q_xx
    @property
    def Q_xy(self): return self._Q_xy
    @property
    def Q_yx(self): return self._Q_yx
    @property
    def Q_yy(self): return self._Q_yy

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------
    def __repr__(self):
        return f"<ParamPKF(dim_y={self.dim_y}, dim_x={self.dim_x}, verbose={self.verbose})>"

    def summary(self):
        print("=== ParamPKF Summary ===")
        print(f"Dimension des observations (dim_y) : {self.dim_y}")
        print(f"Dimension des états (dim_x)        : {self.dim_x}")
        print(f"Taille attendue des matrices       : {self.n} x {self.n}")
        print("Matrice A :")
        print(self.A)
        print("Bloc A_xx :\n", self.A_xx)
        print("Bloc A_xy :\n", self.A_xy)
        print("Bloc A_yx :\n", self.A_yx)
        print("Bloc A_yy :\n", self.A_yy)
        print("Matrice Q :")
        print(self.mQ)
        print("Bloc Q_xx :\n", self.Q_xx)
        print("Bloc Q_xy :\n", self.Q_xy)
        print("Bloc Q_yx :\n", self.Q_yx)
        print("Bloc Q_yy :\n", self.Q_yy)
        print(f"Verbose : {self.verbose}")
        print("========================")
        self._check_consistency()

    # ------------------------------------------------------------------
    # Méthodes privées pour ActiveView
    # ------------------------------------------------------------------
    def _update_A_views(self):
        self._A_xx = ActiveView(self._A, slice(0,self.dim_x),       slice(0,self.dim_x),       self._check_consistency)
        self._A_xy = ActiveView(self._A, slice(0,self.dim_x),       slice(self.dim_x,self._n), self._check_consistency)
        self._A_yx = ActiveView(self._A, slice(self.dim_x,self._n), slice(0,self.dim_x),       self._check_consistency)
        self._A_yy = ActiveView(self._A, slice(self.dim_x,self._n), slice(self.dim_x,self._n), self._check_consistency)

    def _update_Q_views(self):
        self._Q_xx = ActiveView(self._mQ, slice(0,self.dim_x),       slice(0,self.dim_x),       self._check_consistency)
        self._Q_xy = ActiveView(self._mQ, slice(0,self.dim_x),       slice(self.dim_x,self._n), self._check_consistency)
        self._Q_yx = ActiveView(self._mQ, slice(self.dim_x,self._n), slice(0,self.dim_x),       self._check_consistency)
        self._Q_yy = ActiveView(self._mQ, slice(self.dim_x,self._n), slice(self.dim_x,self._n), self._check_consistency)

    # ------------------------------------------------------------------
    # Vérification de cohérence et PSD des blocs
    # ------------------------------------------------------------------
    def _check_consistency(self):
        """Vérifie la cohérence de A, mQ et que _Q_xx/_Q_yy sont PSD"""
        # eig_A = np.linalg.eigvals(self.A)
        # if np.any(np.abs(eig_A) > 1):
        #     print("[Warning] A n'est pas stable (valeurs propres > 1 en module)")

        if not np.allclose(self.mQ, self.mQ.T, atol=1e-12):
            print("[Warning] mQ n'est pas symétrique")

        eig_Q = np.linalg.eigvals(self.mQ)
        if np.any(eig_Q < -1e-12):
            print("[Warning] mQ n'est pas positive semi-définie")

        # Vérification des blocs carrés de covariance
        for name, Q_block in [("_Q_xx", self._Q_xx), ("_Q_yy", self._Q_yy)]:
            # Accès au tableau numpy réel
            Q_block_array = Q_block._parent[Q_block._rows, Q_block._cols]
            
            if Q_block_array.shape[0] != Q_block_array.shape[1]:
                raise ValueError(f"{name} n'est pas carré : {Q_block_array.shape}")
            eig_block = np.linalg.eigvals(Q_block_array)
            if np.any(eig_block < -1e-12):
                print(f"[Warning] {name} n'est pas positive semi-définie")
            if self.verbose >= 2:
                print(f"Valeurs propres de {name} : {eig_block}")

        if self.verbose >= 2:
            print(f"Valeurs propres de A : {eig_A}")
            print(f"Valeurs propres de Q : {eig_Q}")


if __name__ == '__main__':
    
    """
    python prg/classes/ParamPKF.py
    """

    # ------------------------------------------------------
    # Définition des dimensions
    # ------------------------------------------------------
    dim_x = 2  # dimension des états
    dim_y = 2  # dimension des observations
    n = dim_x + dim_y

    # ------------------------------------------------------
    # Création de matrices A et mQ
    # ------------------------------------------------------
 
    A = np.array([[0.5, 0.0, 0.1, 0.0],
                  [0.0, 0.8, 0.0, 0.2],
                  [0.1, 0.2, 1.0, 0.0],
                  [0.0, 0.1, 0.0, 0.9]])

    a, b, c, d, e = 0.1, 0.3, 0.1, 0.2, 0.1
    mQ = np.array([[1.0, b,   a,   d]  ,
                   [b,   1.0, e,   c  ],
                   [a,   e,   1.0, b  ],
                   [d,   c,   b,   1.0]])

    # ------------------------------------------------------
    # Création de l'objet ParamPKF
    # ------------------------------------------------------
    param = ParamPKF(dim_y=dim_y, dim_x=dim_x, A=A, mQ=mQ, verbose=1)

    # ------------------------------------------------------
    # Affichage résumé
    # ------------------------------------------------------
    param.summary()

    # ------------------------------------------------------
    # Modification via les blocs (vues)
    # ------------------------------------------------------
    print("\n--- Modification via les blocs ---")
    param.A_xx[0,1] = 0.99       # modifie A directement
    param.A_xy[1,0] = -0.5       # modifie A directement
    param.Q_yy[0,1] = 0.25       # modifie mQ directement

    # ------------------------------------------------------
    # Affichage après modification
    # ------------------------------------------------------
    print("\nA après modification via les blocs :\n", param.A)
    print("mQ après modification via les blocs :\n", param.mQ)

    # ------------------------------------------------------
    # Remplacement complet via setters
    # ------------------------------------------------------
    new_A = np.eye(n) * 0.9
    param.A = new_A
    print("\nA après remplacement complet via setter :\n", param.A)

    new_Q = np.eye(n) * 0.5
    param.mQ = new_Q
    print("\nmQ après remplacement complet via setter :\n", param.mQ)