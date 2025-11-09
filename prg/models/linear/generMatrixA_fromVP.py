import numpy as np

if __name__ == "__main__":
    # Dimension de la matrice
    n = 2

    # Générer une matrice aléatoire de vecteurs propres
    Q, _ = np.linalg.qr(np.random.randn(n, n))  # matrice orthogonale

    # Créer une matrice diagonale avec au moins une valeur propre > 1
    eigenvalues = np.array([1.2, 0.5])  # 1.2 > 1
    D = np.diag(eigenvalues)

    # Construire la matrice finale : A = Q D Q^T
    A = Q @ D @ Q.T

    # Vérification des valeurs propres
    vals = np.linalg.eigvals(A)
    print("Matrice A :\n", A)
    print("Valeurs propres :", vals)
    print("Au moins une valeur propre>1 :", np.any(vals > 1))
