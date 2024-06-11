from gurobipy import Model, GRB, MVar
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters

def solve(Np, x: npt.NDArray, w: npt.NDArray, A: npt.NDArray, B: npt.NDArray, C: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
    # Params
    M = 1000
    sigma = 6500 # [N]
    kappa = 7000 # [Ns/m]

    # Number of states (n) and inputs (m) and constraints (ncon)
    n = A.shape[0]
    m = B.shape[1]

    A_tilde = np.vstack([np.linalg.matrix_power(A, i+1) @ C for i in range(Np)])
    B_tilde = np.zeros((n*Np, m*Np))
    Q_tilde = np.zeros((n*Np, n*Np))
    R_tilde = np.zeros((m*Np, m*Np))
    for i in range(Np):
        for j in range(i+1):
            B_tilde[i*n:(i+1)*n, j*m:(j+1)*m] = C @ np.linalg.matrix_power(A, i-j) @ B
        Q_tilde[i * n: (i + 1) * n, i * n: (i + 1) * n] = Q
        R_tilde[i * m: (i + 1) * m, i * m: (i + 1) * m] = R
    
    H = B_tilde.T @ Q_tilde @ B_tilde + R_tilde # checked
    f = 2 * x.T @ A_tilde.T @ Q_tilde @ B_tilde # checked

    model = Model('MPC controller')
    model.Params.LogToConsole = 0
    
    w_tilde = dict()
    f_tilde = dict()
    w = np.reshape(w, (Np, 1))
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
        delta[i] = model.addVar(vtype=GRB.BINARY, name='delta[{}]'.format(i))

    model.update()
    for i in range(Np):
        model.addConstr(u_tilde[i*2 + 1, 0] <= sigma)
        model.addConstr(u_tilde[i*2 + 1, 0] >= -sigma)
        # Might need opposite to froce delta to 0
        ########### model.addGenConstrIndicator(delta[i], 1, A_tilde[i*n: i*n+n, :] @ x + B_tilde[i*n: i*n+n, 0:i*m+m] @ u_tilde[0:i*m+m], GRB.GREATER_EQUAL, 0)
        model.addConstr(A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m] >= 1e-8 -M * (1 - delta[i]))
        model.addConstr(A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m] <= M * delta[i])
        model.addGenConstrIndicator(delta[i], 1, u_tilde[i*2 + 1, 0], GRB.GREATER_EQUAL, 1e-8)
        model.addGenConstrIndicator(delta[i], 0, u_tilde[i*2 + 1, 0], GRB.LESS_EQUAL, 0)
        model.addConstr(u_tilde[i*2 + 1, 0] - kappa * (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]) - 2*delta[i]*(u_tilde[i*2 + 1, 0] - kappa * (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m])) >= 0)

    obj = u_tilde.T @ H @ u_tilde + f @ u_tilde
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    # model.write('model.lp')
    model.optimize()

    if model.status == GRB.OPTIMAL:
        u = []
        for i in range(Np):
            u.append(model.getVarByName(f'f_tilde[{i}]').X)
        return u
    
def quarter_car(par: Parameters, Np:int, dt: float, x: npt.NDArray, wf: npt.NDArray, wb: npt.NDArray):
    #   state
    #       1 zu-zr
    #       2 zu'
    #       3 zs-zu
    #       4 zs'
    #   input
    #       1 zr
    #       2 f
    dr = 0.7
    Af = np.array([
        [0, 1, 0, 0],
        [-par.ktf/par.muf, -dr/par.muf, par.ksf/par.muf, dr/par.muf],
        [0, -1, 0, 1],
        [0, dr/par.ms/2, -par.ksf/par.ms/2, -dr/par.ms/2]
    ])

    Bf = np.array([
        [0, 0],
        [par.ktf/par.muf, par.ms/par.muf],
        [0, 0],
        [0, -1]
    ])

    Cf = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, dr/par.ms/2, -par.ksf/par.ms/2, -dr/par.ms/2]])

    # Cf = np.eye(4)

    Ab = np.array([
        [0, 1, 0, 0], 
        [-par.ktr/par.mur, -dr/par.mur, par.ksr/par.mur, dr/par.mur], 
        [0, -1, 0, 1], 
        [0, dr/par.ms/2, -par.ksr/par.ms/2, -dr/par.ms/2]
    ])

    Bb = np.array([
        [0, 0], 
        [par.ktr/par.mur, par.ms/par.mur], 
        [0, 0], 
        [0, -1]
    ])

    Cb = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, dr/par.ms/2, -par.ksr/par.ms/2, -dr/par.ms/2]])

    # Cb = np.eye(4)

    Q = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]])

    ssf = signal.cont2discrete((Af, Bf, Cf, np.zeros((4, 2))), dt)
    ssb = signal.cont2discrete((Ab, Bb, Cb, np.zeros((4, 2))), dt)

    Af = ssf[0]
    Bf = ssf[1]
    Ab = ssb[0]
    Bb = ssb[1]

    uf = solve(Np, np.array([[x[4, 0]], [x[5, 0]], [x[0, 0]], [x[1, 0]]]), wf, Af, Bf, Cf, Q, np.zeros((2, 2)))
    ub = solve(Np, np.array([[x[6, 0]], [x[7, 0]], [x[2, 0]], [x[3, 0]]]), wb, Ab, Bb, Cb, Q, np.zeros((2, 2)))
    return uf[0], ub[0]


if __name__ == "__main__":
    quarter_car(par = Parameters(960, 1222, 40, 45, 200000,
                    200000, 18000, 22000, 1000,
                    1000, 1.3, 1.5),
                Np=10, 
                dt=0.01, 
                x=np.array([[0], [0.1], [0], [0.1], [0], [0.1], [0], [0.1]]), 
                wfdot=np.zeros((10, 1)),
                wbdot=np.zeros((10, 1))
               )