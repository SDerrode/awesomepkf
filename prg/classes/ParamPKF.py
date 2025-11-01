#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

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
    Classe pour stocker les paramètres d'un filtre de Kalman couple.
    """

    def __init__(self, dim_y, dim_x, A, mQ, verbose=0):
        # Vérifications des dimensions
        if not isinstance(dim_y, int) or dim_y <= 0:
            raise ValueError("dim_y doit être un entier > 0")
        if not isinstance(dim_x, int) or dim_x <= 0:
            raise ValueError("dim_x doit être un entier > 0")
        self.dim_y  = dim_y
        self.dim_x  = dim_x
        self.dim_xy = dim_x + dim_y
        
        # Verbose
        if verbose not in [0,1,2]:
            raise ValueError("verbose doit être 0, 1 ou 2")
        self.verbose = verbose

        # Système de paramètres A, mQ
        ###################################

        # Vérification des matrices A et mQ
        self._A = np.array(A)
        if self._A.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"A doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._update_A_views()

        self._mQ = np.array(mQ)
        if self._mQ.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"mQ doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._update_Q_views()
        
        self._mu0 = np.zeros(self.dim_xy)
        
        # Système de paramètres Sigma, (Q1, Q2)
        self._updateSigma()

        # Vérification initiale
        self._check_consistency()


    def _updateSigma(self):
        """Met à jour Q1, Q2 et Sigma à partir de A et mQ, 
        puis vérifie la relation Q ≈ Q1 - A Q2^T."""

        # --- Résolution de l'équation de Lyapunov : Q1 - A Q1 A^T = Q ---
        self._Q1 = solve_discrete_lyapunov(self._A, self._mQ)

        # --- Calcul de Q2 = A Q1 ---
        self._Q2 = self._A @ self._Q1

        # --- Construction de Sigma = [[Q1, Q2^T], [Q2, Q1]] ---
        self._Sigma = np.block([
            [self._Q1, self._Q2.T],
            [self._Q2, self._Q1]
        ])

        # --- Vérification de la cohérence Q ≈ Q1 - A Q2^T ---
        Q_est     = self._Q1 - self._A @ self._Q2.T
        diff      = self._mQ - Q_est
        norm_diff = np.linalg.norm(diff)
        norm_ref  = np.linalg.norm(self._mQ)

        rel_error = norm_diff / (norm_ref + 1e-12)
        if rel_error > 1e-8:
            warnings.warn(
                f"[ParamPKF] Incohérence : Q ≉ Q1 - A Q2^T "
                f"(erreur relative = {rel_error:.2e})",
                UserWarning
            )
            if self.verbose >= 2:
                print(f"Différence Q - (Q1 - A Q2^T) =\n{diff}")
        elif self.verbose >= 2:
            print(f"[ParamPKF] Vérification OK : ||Q - (Q1 - A Q2^T)||_rel = {rel_error:.2e}")

        # creation des vues a, b, c, d et e
        self._a = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x  , 0:self.dim_x]
        self._b = self._Sigma[self.dim_x:self.dim_xy              , 0:self.dim_x]
        self._c = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, self.dim_x:self.dim_xy]
        self._d = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, 0:self.dim_x]
        self._e = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x  , self.dim_x:self.dim_xy]
        # creation des vues sur les block diagonaux
        self._Sigma_X1 = self._Sigma[0:self.dim_x,                         0:self.dim_x]
        self._Sigma_Y1 = self._Sigma[self.dim_x:self.dim_xy,               self.dim_x:self.dim_xy]
        self._Sigma_X2 = self._Sigma[self.dim_xy:self.dim_xy+self.dim_x,   self.dim_xy:self.dim_xy+self.dim_x]
        self._Sigma_Y2 = self._Sigma[self.dim_xy+self.dim_x:2*self.dim_xy, self.dim_xy+self.dim_x:2*self.dim_xy]
        

    # ------------------------------------------------------------------
    # Propriétés et setters pour A
    # ------------------------------------------------------------------
    @property
    def A(self): 
        return self._A

    @A.setter
    def A(self, new_A):
        new_A = np.array(new_A)
        if new_A.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"A doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._A = new_A
        self._update_A_views()
        self._updateSigma()
        if self.verbose >= 1:
            print("[ParamPKF] A mis à jour")
        self._check_consistency()


    # ------------------------------------------------------------------
    # Propriétés et setters pour mQ
    # ------------------------------------------------------------------
    @property
    def mQ(self): 
        return self._mQ

    @mQ.setter
    def mQ(self, new_Q):
        new_Q = np.array(new_Q)
        if new_Q.shape != (self.dim_xy, self.dim_xy):
            raise ValueError(f"mQ doit être carrée de dimension ({self.dim_xy},{self.dim_xy})")
        self._mQ = new_Q
        self._update_Q_views()
        self._updateSigma() 
        if self.verbose >= 1:
            print("[ParamPKF] mQ mis à jour")
        self._check_consistency()


    # ------------------------------------------------------------------
    # Propriété pour n
    # ------------------------------------------------------------------
    @property
    def n(self): return self.dim_xy

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
    # Getters pour les matrices sur Sigma et Q1, Q2, et les vues a, b, c, d et e
    # ------------------------------------------------------------------
    @property
    def Sigma(self): return self._Sigma
    @property
    def Q1(self): return self._Q1
    @property
    def mu0(self): return self._mu0
    @property
    def Q2(self): return self._Q2
    @property
    def a(self): return self._a
    @property
    def b(self): return self._b
    @property
    def c(self): return self._c
    @property
    def d(self): return self._d
    @property
    def e(self): return self._e
    @property
    def Sigma_X1(self): return self._Sigma_X1
    @property
    def Sigma_Y1(self): return self._Sigma_Y1
    @property
    def Sigma_X2(self): return self._Sigma_X2
    @property
    def Sigma_Y2(self): return self._Sigma_Y2

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------
    def __repr__(self):
        return f"<ParamPKF(dim_y={self.dim_y}, dim_x={self.dim_x}, verbose={self.verbose})>"

    def summary(self):
        """Affiche un résumé complet des matrices et blocs, avec 2 chiffres après la virgule."""

        def fmt(M):
            """Formatage pour afficher matrices avec 2 chiffres après la virgule, compatible ActiveView."""
            if hasattr(M, "_parent"):  # ActiveView
                M = M._parent[M._rows, M._cols]
            return np.array2string(M, formatter={'float_kind':lambda x: f"{x:.2f}"})

        print(f"=== ParamPKF Summary ===")
        print(f"Dimension des observations (dim_y) : {self.dim_y}")
        print(f"Dimension des états (dim_x)        : {self.dim_x}")
        print(f"Taille attendue des matrices A et Q: {self.dim_xy} x {self.dim_xy}\n")

        print(f"Matrice A :\n{fmt(self.A)}")
        print(f"Bloc A_xx :\n{fmt(self.A_xx)}")
        print(f"Bloc A_xy :\n{fmt(self.A_xy)}")
        print(f"Bloc A_yx :\n{fmt(self.A_yx)}")
        print(f"Bloc A_yy :\n{fmt(self.A_yy)}\n")

        print(f"Matrice Q :\n{fmt(self.mQ)}")
        print(f"Bloc Q_xx :\n{fmt(self.Q_xx)}")
        print(f"Bloc Q_xy :\n{fmt(self.Q_xy)}")
        print(f"Bloc Q_yx :\n{fmt(self.Q_yx)}")
        print(f"Bloc Q_yy :\n{fmt(self.Q_yy)}\n")

        print(f"Matrice Sigma :\n{fmt(self._Sigma)}")
        print(f"Sigma_X1 :\n{fmt(self._Sigma_X1)}")
        print(f"Sigma_Y1 :\n{fmt(self._Sigma_Y1)}")
        print(f"Sigma_X2 :\n{fmt(self._Sigma_X2)}")
        print(f"Sigma_Y2 :\n{fmt(self._Sigma_Y2)}")
        print(f"Q1 :\n{fmt(self._Q1)}")
        print(f"Q2 :\n{fmt(self._Q2)}")
        print(f"mu0 :\n{fmt(self._mu0)}\n")

        print("Vues sur Sigma :")
        for name in ['_a', '_b', '_c', '_d', '_e']:
            if hasattr(self, name):
                print(f"{name[1:]} =\n{fmt(getattr(self, name))}")

        print(f"\nVerbose : {self.verbose}")
        print("========================")




    # ------------------------------------------------------------------
    # Méthodes privées pour ActiveView
    # ------------------------------------------------------------------
    def _update_A_views(self):
        def _callback():
            self._updateSigma()
            self._check_consistency()
        self._A_xx = ActiveView(self._A, slice(0,self.dim_x),       slice(0,self.dim_x),       _callback)
        self._A_xy = ActiveView(self._A, slice(0,self.dim_x),       slice(self.dim_x,self.dim_xy), _callback)
        self._A_yx = ActiveView(self._A, slice(self.dim_x,self.dim_xy), slice(0,self.dim_x),       _callback)
        self._A_yy = ActiveView(self._A, slice(self.dim_x,self.dim_xy), slice(self.dim_x,self.dim_xy), _callback)

    def _update_Q_views(self):
        def _callback():
            self._updateSigma()
            self._check_consistency()
        self._Q_xx = ActiveView(self._mQ, slice(0,self.dim_x),           slice(0,self.dim_x),           _callback)
        self._Q_xy = ActiveView(self._mQ, slice(0,self.dim_x),           slice(self.dim_x,self.dim_xy), _callback)
        self._Q_yx = ActiveView(self._mQ, slice(self.dim_x,self.dim_xy), slice(0,self.dim_x),           _callback)
        self._Q_yy = ActiveView(self._mQ, slice(self.dim_x,self.dim_xy), slice(self.dim_x,self.dim_xy), _callback)


    # ------------------------------------------------------------------
    # Vérification de cohérence et PSD des blocs
    # ------------------------------------------------------------------
    def _check_consistency(self):
        """Vérifie la cohérence interne des matrices (A, mQ, Q1, Sigma)."""

        # --- Fonction utilitaire locale ---
        def _is_covariance(M, name, atol=1e-12):
            """Vérifie si une matrice est symétrique positive semi-définie."""
            if M is None:
                return
            if not np.allclose(M, M.T, atol=atol):
                print(f"[Warning] {name} n'est pas symétrique")
            eigvals = np.linalg.eigvals(M)
            if np.any(eigvals < -atol):
                print(f"[Warning] {name} n'est pas positive semi-définie (valeurs propres min = {eigvals.min():.3e})")
            if self.verbose >= 2:
                print(f"Valeurs propres de {name} : {eigvals}")
            return eigvals

        # --- Vérification de mQ ---
        _is_covariance(self.mQ, "mQ")

        # --- Vérification des blocs de mQ (Q_xx, Q_yy) ---
        for name, Q_block in [("_Q_xx", self._Q_xx), ("_Q_yy", self._Q_yy)]:
            Q_block_array = Q_block._parent[Q_block._rows, Q_block._cols]
            if Q_block_array.shape[0] != Q_block_array.shape[1]:
                raise ValueError(f"{name} n'est pas carré : {Q_block_array.shape}")
            _is_covariance(Q_block_array, name)

        # --- Vérification de Q1 (si présent) ---
        if hasattr(self, "_Q1"):
            _is_covariance(self._Q1, "_Q1")

        # --- Vérification de Sigma (si présente) ---
        if hasattr(self, "_Sigma"):
            _is_covariance(self._Sigma, "_Sigma")

        # --- Vérification de la stabilité de A ---
        # eig_A = np.linalg.eigvals(self.A)
        # if np.any(np.abs(eig_A) > 1 + 1e-12):
        #     print("[Warning] A n'est pas stable (|λ| > 1)")
        # if self.verbose >= 2:
        #     print(f"Valeurs propres de A : {eig_A}")


if __name__ == '__main__':
    
    """
    python prg/classes/ParamPKF.py
    """

    # ------------------------------------------------------
    # Exemples de jeux de paramètres
    # ------------------------------------------------------
    
    # exemple #1
    dim_x  = 2  # dimension des états
    dim_y  = 2  # dimension des observations
    dim_xy = dim_x + dim_y

    A = np.array([[5, 2, 1, 0],
                  [3, 8, 0, 2],
                  [2, 2, 10, 6],
                  [1, 1, 5, 9]], dtype=float)

    a, b, c, d, e = 0.1, 0.5, 0.1, 0.2, 0.1
    mQ = np.array([[1.0,  b,   a,   d]  ,
                    [b,   1.0, e,   c  ],
                    [a,   e,   1.0, b  ],
                    [d,   c,   b,   1.0]]) *3.
    
    # exemple #2
    # dim_x  = 1  # dimension des états
    # dim_y  = 1  # dimension des observations
    # dim_xy = dim_x + dim_y

    # A = np.array([[0.8, 0.1],
    #             [0.0, 0.9]])
    # mQ = np.array([[1.0, 0.2],
    #             [0.2, 1.0]])

    

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
    # print("\n--- Modification via les blocs ---")
    # param.A_xx[0,0] = 0.99       # modifie A directement
    # param.A_xy[0,0] = -0.5       # modifie A directement
    # param.Q_yy[0,0] = 0.25       # modifie mQ directement

    # # ------------------------------------------------------
    # # Affichage après modification
    # # ------------------------------------------------------
    # print(f"\nA après modification via les blocs :\n{param.A}")
    # print(f"mQ après modification via les blocs :\n{param.mQ}")

    # # ------------------------------------------------------
    # # Remplacement complet via setters
    # # ------------------------------------------------------
    # new_A = np.eye(dim_xy) * 0.9
    # param.A = new_A
    # print(f"\nA après remplacement complet via setter :\n{param.A}")

    # new_Q = np.eye(dim_xy) * 0.5
    # param.mQ = new_Q
    # print(f"\nmQ après remplacement complet via setter :\n{param.mQ}")