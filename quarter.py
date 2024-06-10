from gurobipy import Model, GRB
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def solve(Np, x: npt.NDArray, A: npt.NDArray, B: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
    # Params
    sigma = 6500 # [N]

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

    xk = A_tilde @ x + B_tilde @ np.ones((m*Np, 1))
    xk2 = []
    for i in range(Np):
        xk2.append(A_tilde[i*n: i*n+n, :] @ x + B_tilde[i*n: i*n+n, 0:i*m+m] @ np.ones((m, 1)))

    print(xk)
    print(xk2)
    
    H = B_tilde.T @ Q_tilde @ B_tilde + R_tilde # checked
    f = 2 * x.T @ A_tilde.T @ Q_tilde @ B_tilde # checked

    model = Model('MPC controller')
    model.Params.LogToConsole = 0
    
    u_tilde = model.addMVar(shape=(m*Np), vtype=GRB.CONTINUOUS, name='u_tilde', lb=-GRB.INFINITY, ub=GRB.INFINITY)
    delta = dict()
    for i in range(Np):
        delta[i] = model.addVar(vtype=GRB.BINARY, name='delta_v[{}]'.format(i))

    model.update()
    for i in range(Np):
        model.addConstr(u_tilde[i] <= sigma)
        model.addConstr(u_tilde[i] >= -sigma)
        model.addGenConstrIndicator(delta[i], 1, A_tilde[i*n: i*n+n, :] @ x + B_tilde[i*n: i*n+n, 0:i*m+m] @ u_tilde[i] <= sigma)

    obj = u_tilde.T @ H @ u_tilde + f @ u_tilde
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.getVarByName('u_tilde[0]').X
    

A = np.array([[1, 0.1], [0, 1]])
B = np.array([[0], [0.1]])
Q = np.eye(2)
R = np.array([1])
x = np.array([[10], [0]])
solve(10, x, A, B, Q, R)
# x1result = []
# x2result = []
# uresult = []
# for _ in range(160):
#     uk = solve(10, x, A, B, Q, R)
#     x = A@x + B*uk
#     x1result.append(x[0])
#     x2result.append(x[1])
#     uresult.append(uk)

# plt.plot(x1result)
# plt.plot(x2result)
# plt.plot(uresult)
# plt.grid()
# plt.show()

