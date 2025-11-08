#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class UKFSimulator:
    def __init__(self, f, h, Q, R, x0, P0, alpha=1e-3, beta=2, kappa=0):
        """
        f      : fonction de transition d'état x_{k+1} = f(x_k)
        h      : fonction d'observation y_k = h(x_k)
        Q, R   : covariances de bruit
        x0, P0 : état et covariance initiale
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.n = len(x0)

        # Paramètres UKF
        self.alpha   = alpha
        self.beta    = beta
        self.kappa   = kappa
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.gamma   = np.sqrt(self.n + self.lambda_)

        # Poids moyenne Wm, et poids correlation Wc
        self.Wm    = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc    = np.copy(self.Wm)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)

    def sigma_points(self, x, P):
        """Génère les 2n+1 sigma-points autour de x"""
        A = np.linalg.cholesky(P)
        sigma = [x]
        for i in range(self.n):
            sigma.append(x + self.gamma * A[:, i])
            sigma.append(x - self.gamma * A[:, i])
        return np.array(sigma)

    def predict(self):
        """Étape de prédiction du UKF"""
        sigma = self.sigma_points(self.x, self.P)
        X_pred = np.array([self.f(s) for s in sigma])
        x_pred = np.sum(self.Wm[:, None] * X_pred, axis=0)
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            y = X_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)
        self.x, self.P = x_pred, P_pred
        self.sigma_pred = X_pred

    def update(self, y_meas):
        """Étape de mise à jour du UKF"""
        Z_pred = np.array([self.h(s) for s in self.sigma_pred])
        z_mean = np.sum(self.Wm * Z_pred)
        P_zz = self.R.copy()
        P_xz = np.zeros((self.n, 1))
        for i in range(2 * self.n + 1):
            dz = Z_pred[i] - z_mean
            dx = self.sigma_pred[i] - self.x
            P_zz += self.Wc[i] * dz * dz
            P_xz += self.Wc[i] * np.outer(dx, dz)
        K = P_xz / P_zz
        self.x += (K.flatten()) * (y_meas - z_mean)
        self.P -= K @ P_zz @ K.T

    def step(self, y_meas):
        """Une étape complète prédiction + mise à jour"""
        self.predict()
        self.update(y_meas)
        return self.x

    def simulate(self, N):
        """Simule un jeu de données non linéaires"""
        x_true = np.zeros((N, self.n))
        y_obs = np.zeros(N)
        x = self.x.copy()
        for k in range(N):
            w = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            v = np.random.normal(0, np.sqrt(self.R))
            x = self.f(x) + w
            y = self.h(x) + v
            x_true[k] = x
            y_obs[k] = y
        return x_true, y_obs

    def test_covariance(self):
        """Vérifie que P est bien définie positive"""
        eig = np.linalg.eigvals(self.P)
        if np.any(eig <= 0):
            print("⚠️ Covariance non dPSD...")
            self.P = (self.P + self.P.T) / 2
            min_eig = np.min(np.real(eig))
            self.P += np.eye(self.n) * abs(min_eig) * 1.01

# Nouveau modèle non linéaire (exemple radar-like)


if __name__ == "__main__":

    # Paramètres
    Q  = np.diag([1e-3, 1e-3])
    R  = np.array([[1e-2]])
    x0 = np.array([0.5, 0.2])
    P0 = np.eye(2)

    ukf = UKFSimulator(f, h, Q, R, x0, P0)

    # Simulation
    x_true, y_obs = ukf.simulate(100)
    print(f'x_true={x_true}')

    # Filtrage
    x_filt = np.zeros_like(x_true)
    for k in range(len(y_obs)):
        ukf.step(y_obs[k])
        ukf.test_covariance()
        x_filt[k] = ukf.x

    # Visualisation
    plt.figure(figsize=(10,4))
    plt.plot(x_true[:,0], label="x1 vrai")
    plt.plot(x_filt[:,0], label="x1 estimé", linestyle='--')
    plt.xlabel("Temps k"); plt.ylabel("État")
    plt.legend(); plt.title("UKF sur système non linéaire")
    plt.show()
