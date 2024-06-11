from gurobipy import Model, GRB, MVar
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters

def solve(Np, x: npt.NDArray, w: npt.NDArray, A: npt.NDArray, B: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
    # Params
    sigma = 6500 # [N]
    zmin = -1000
    zmax = 1000

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
    
    H = B_tilde.T @ Q_tilde @ B_tilde + R_tilde # checked
    f = 2 * x.T @ A_tilde.T @ Q_tilde @ B_tilde # checked

    model = Model('MPC controller')
    #model.Params.LogToConsole = 0
    
    w_tilde = dict()
    f_tilde = dict()
    for i in range(Np):
        w_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'w_tilde[{i}]', lb=w[i, 0]-1e-8, ub=w[i, 0]+1e-8)
        f_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'f_tilde[{i}]', lb=-GRB.INFINITY, ub=GRB.INFINITY)

    u_list = []
    for i in range(Np):
        u_list.append([w_tilde[i]])
        u_list.append([f_tilde[i]])

    u_tilde = MVar.fromlist(u_list)

    delta = dict()
    for i in range(Np):
        delta[i] = model.addVar(vtype=GRB.BINARY, name='delta_v[{}]'.format(i))

    model.update()
    for i in range(Np):
        model.addConstr(u_tilde[i*2 + 1, 0] <= sigma)
        model.addConstr(u_tilde[i*2 + 1, 0] >= -sigma)
        # Might need opposite to froce delta to 0
        model.addGenConstrIndicator(delta[i], 1, A_tilde[i*n: i*n+n, :] @ x + B_tilde[i*n: i*n+n, 0:i*m+m] @ u_tilde[0:i*m+m], GRB.GREATER_EQUAL, 0)
        model.addGenConstrIndicator(delta[i], 1, A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                                               - A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m], 
                                                 GRB.GREATER_EQUAL, 0)
        model.addConstr(A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m] <= zmax * (1 - delta[i]))
        model.addConstr(A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m] >= (zmin-1e-8) * delta[i] - 1e-8)

    obj = u_tilde.T @ H @ u_tilde + f @ u_tilde
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    model.write('model.lp')
    model.optimize()

    if model.status == GRB.OPTIMAL:
        xk = []
        for i in range(Np):
            xk.append(model.getVarByName(f'f_tilde[{i}]').X)
        return xk
    
def quarter_car(par: Parameters, dt: float):
    #   state
    #       1 zu-zr
    #       2 zu'
    #       3 zs-zu
    #       4 zs'
    #   input
    #       1 zr
    #       2 f
    
    Af = np.array([
        [0, 1, 0, 0],
        [-par.ktf/par.muf, -par.csf/par.muf, par.ksf/par.muf, par.csf/par.muf],
        [0, -1, 0, 1],
        [0, par.csf/par.ms, -par.ksf/par.ms, -par.csf/par.ms]
    ])

    Bf = np.array([
        [0, 0],
        [par.ktf/par.muf, par.ms/par.muf],
        [0, 0],
        [0, -1]
    ])

    ss = signal.cont2discrete((Af, Bf, np.eye(4), np.zeros((4, 2))), dt)

    Af = ss[0]
    Bf = ss[1]

    uk = solve(10, np.array([[0], [0.1], [0], [0.1]]), np.ones((10, 1))*0.0001, Af, Bf, np.eye(4), np.eye(2))

# A = np.array([[1, 0.1], [0, 1]])
# B = np.array([[0], [0.1]])
# Q = np.eye(2)
# R = np.array([1])
# x = np.array([[10], [0]])
# x1result = []
# x2result = []
# uresult = []
# for _ in range(160):
#     uk = solve(10, x, np.zeros(1), A, B, Q, R)
#     x = A@x + B*uk
#     x1result.append(x[0])
#     x2result.append(x[1])
#     uresult.append(uk)

# plt.plot(x1result)
# plt.plot(x2result)
# plt.plot(uresult)
# plt.grid()
# plt.show()

quarter_car(par = Parameters(960, 1222, 40, 45, 200000,
                 200000, 18000, 22000, 1000,
                 1000, 1.3, 1.5))