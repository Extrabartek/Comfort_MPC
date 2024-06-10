from gurobipy import Model, GRB
import numpy as np
import numpy.typing as npt

def solve(Np, x0: npt.NDArray, A: npt.NDArray, B: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
    # Number of states (n) and inputs (m) and constraints (ncon)
    n = A.shape[0]
    m = B.shape[1]

    A_tilde = np.vstack([np.linalg.matrix_power(A, i+1) for i in range(Np)])
    B_tilde = np.zeros((n*Np, m*Np))
    Q_tilde = np.zeros((n*Np, n*Np))
    R_tilde = np.zeros((m*Np, m*Np))
    for i in range(Np):
        for j in range(i+1):
            B_tilde[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j).dot(B)
        Q_tilde[i * n: (i + 1) * n, i * n: (i + 1) * n] = Q
        R_tilde[i * m: (i + 1) * m, i * m: (i + 1) * m] = R
    
    H = B_tilde.T @ Q_tilde @ B_tilde + R_tilde
    f = 2 * x0.T @ A_tilde.T @ Q_tilde @ B_tilde

    model = Model()
    
    u = dict()
    for i in range(Np):
        for j in range(m):
            u[j, i] = model.addVar(vtype=GRB.BINARY, name=f'u_{j}_{i}')
    model.update()

    obj = 0
    for i in range(Np):
        for j in range(m):
            obj += H[j, i]*u[j, i]*u[j, i] + f[]

