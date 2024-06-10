import numpy as np
import numpy.typing as npt

def create_matrices(Np: int, A: npt.NDArray, B: npt.NDArray, Q: npt.NDArray, R: npt.NDArray, E: npt.NDArray, F: npt.NDArray, G: npt.NDArray):
    # Number of states (n) and inputs (m) and constraints (ncon)
    n = A.shape[0]
    m = B.shape[1]
    ncon = G.shape[0]

    A_tilde = np.vstack([np.linalg.matrix_power(A, i+1) for i in range(Np)])
    B_tilde = np.zeros((n*Np, m*Np))
    Q_tilde = np.zeros((n*Np, n*Np))
    R_tilde = np.zeros((m*Np, m*Np))
    E_tilde = np.zeros((ncon*Np, m*Np))
    F_tilde = np.zeros((ncon*Np, n*Np))
    G_tilde = np.vstack([G for _ in range(Np)])
    for i in range(Np):
        for j in range(i+1):
            B_tilde[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j).dot(B)
        Q_tilde[i * n: (i + 1) * n, i * n: (i + 1) * n] = Q
        R_tilde[i * m: (i + 1) * m, i * m: (i + 1) * m] = R
        E_tilde[i * ncon: (i + 1) * ncon, i * m: (i + 1) * m] = E
        F_tilde[i * ncon: (i + 1) * ncon, i * n: (i + 1) * n] = F
    
    return A_tilde, B_tilde, Q_tilde, R_tilde, E_tilde, F_tilde, G_tilde

# # Example usage
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[1], [0]])
# Q = np.array([[1, 0], [0, 1]])
# R = np.array([[1]])
# E = np.array([[1], [-1]])
# F = np.array([[0, 1], [0, -1]])
# G = np.array([[4], [6]])
# Np = 3

# A_tilde, B_tilde, Q_tilde, R_tilde, E_tilde, F_tilde, G_tilde = create_matrices(Np, A, B, Q, R, E, F, G)

# print("A_tilde:")
# print(A_tilde)
# print("\nB_tilde:")
# print(B_tilde)
# print("\nQ_tilde:")
# print(Q_tilde)
# print("\nR_tilde:")
# print(R_tilde)
# print("\nE_tilde:")
# print(E_tilde)
# print("\nF_tilde:")
# print(F_tilde)
# print("\nG_tilde:")
# print(G_tilde)