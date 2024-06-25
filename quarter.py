from gurobipy import Model, GRB, MVar
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters

class SS:
    def __init__(self, par: Parameters, dt: float):
        A = np.array([
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [-par.ksf/(par.ms/2), 0, -par.csf/(par.ms/2), par.csf/(par.ms/2)],
            [par.ksf/par.muf, -par.ktf/par.muf, par.csf/par.muf, -par.csf/par.muf]
        ])
        B = np.array([
            [0, 0],
            [-1, 0],
            [0, -1/(par.ms/2)],
            [0, 1/par.muf]
        ])
        C = np.array([
            [-par.ksf/(par.ms/2), 0, -par.csf/(par.ms/2), par.csf/(par.ms/2)],
            [0, 1, 0, 0]
        ])
        D = np.array([
            [0, -1/(par.ms/2)],
            [0, 0]
        ])

        self.Q = np.array([[1, 0], [0, 1]])

        self.R = np.array([[0, 0], [0, 0]])

        ssd = signal.cont2discrete((A, B, C, D), dt)
        self.A = ssd[0]
        self.B = ssd[1]
        self.C = ssd[2]
        self.D = ssd[3]

        self.par = par

def solve(cs: float, cmin: float, cmax: float, Np, x: npt.NDArray, w: npt.NDArray, A: npt.NDArray, B: npt.NDArray, C: npt.NDArray, D: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
    # Params
    eps = 1e-16
    M = 100000
    # sigma = 6500 # [N]

    print(f"Road profile, MPC: {w}")

    # Number of states (n) and inputs (m) and constraints (ncon)
    n = A.shape[0]
    m = B.shape[1]
    ncon = C.shape[0]

    A_tilde = np.vstack([np.linalg.matrix_power(A, i+1) for i in range(Np)])
    A_c_tilde = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(Np)])
    B_c_tilde = np.zeros((ncon*Np, m*Np))
    B_tilde = np.zeros((n*Np, m*Np))
    D_tilde = np.zeros((ncon*Np, m*Np))
    Q_tilde = np.zeros((ncon*Np, ncon*Np))
    R_tilde = np.zeros((m*Np, m*Np))
    for i in range(Np):
        for j in range(i+1):
            if i-j <= 0:
                B_c_tilde[i*ncon:(i+1)*ncon, j*m:(j+1)*m] = np.zeros((ncon, m))

            else:
                B_c_tilde[i*ncon:(i+1)*ncon, j*m:(j+1)*m] = C @ np.linalg.matrix_power(A, i-j-1) @ B
            if i-j < 0:
                B_tilde[i*n:(i+1)*n, j*m:(j+1)*m] = np.zeros((n, m))
            else:
                B_tilde[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j) @ B
        D_tilde[i * ncon: (i + 1) * ncon, i * m: (i + 1) * m] = D
        Q_tilde[i * ncon: (i + 1) * ncon, i * ncon: (i + 1) * ncon] = Q
        R_tilde[i * m: (i + 1) * m, i * m: (i + 1) * m] = R

    B_c_tilde = B_c_tilde + D_tilde
    H = B_c_tilde.T @ Q_tilde @ B_c_tilde + R_tilde  # checked
    f = 2 * x.T @ A_c_tilde.T @ Q_tilde @ B_c_tilde  # checked

    model = Model('MPC controller')
    model.Params.LogToConsole = 0
    
    w_tilde = dict()
    f_tilde = dict()
    w = np.reshape(w, (Np, 1))
    for i in range(Np):
        w_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'w_tilde[{i}]', lb=w[i, 0]-eps, ub=w[i, 0]+eps)
        f_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'f_tilde[{i}]', lb=-GRB.INFINITY, ub=GRB.INFINITY)

    u_list = []
    for i in range(Np):
        u_list.append([w_tilde[i]])
        u_list.append([f_tilde[i]])

    u_tilde = MVar.fromlist(u_list)

    delta = dict()
    z1 = dict()
    z2 = dict()
    for i in range(Np):
        delta[i] = model.addVar(vtype=GRB.BINARY, name='delta[{}]'.format(i))
        z1[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z1[{}]'.format(i))
        z2[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z2[{}]'.format(i))

    model.update()
    for i in range(Np):
        model.addConstr((A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
                       - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) <= M*delta[i])
        model.addConstr((A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
                       - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) >= -M*(1-delta[i]))

        model.addConstr(z1[i] <= eps + M*delta[i])
        model.addConstr(z1[i] >= -M*delta[i])
        model.addConstr(z1[i] <= eps + u_tilde[i*2 + 1, 0] + (cs - cmin) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                          - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                          + M*(1 - delta[i]))
        model.addConstr(z1[i] >= u_tilde[i*2 + 1, 0] + (cs - cmin) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                    - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                    - M*(1 - delta[i]))
        
        model.addConstr(z2[i] <= eps + M*delta[i])
        model.addConstr(z2[i] >= -M*delta[i])
        model.addConstr(z2[i] <= eps + u_tilde[i*2 + 1, 0] + (cs - cmax) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                          - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                          + M*(1 - delta[i]))
        model.addConstr(z2[i] >= u_tilde[i*2 + 1, 0] + (cs - cmax) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                    - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                    - M*(1 - delta[i]))

        model.addConstr(-u_tilde[i*2 + 1, 0] - (cs - cmin) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                            - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
                                                            + 2*z1[i] >= 0)
        model.addConstr(-u_tilde[i*2 + 1, 0] - (cs - cmax) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                            - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
                                                            + 2*z2[i] <= 0)


    model.update()
    obj = u_tilde.T @ H @ u_tilde + f @ u_tilde
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    # model.write('model.lp')
    model.optimize()

    if model.status == GRB.OPTIMAL:
        u = []
        deltas = []
        deflection_vel_states = A_tilde[2, 0:4] @ x + B_tilde[2, 0:2] * u_tilde[0:2].X \
                              - A_tilde[3, 0:4] @ x + B_tilde[3, 0:2] * u_tilde[0:2].X
        print(f"Deflection velocity states, MPC: {deflection_vel_states}")
        for i in range(Np):
            u.append(model.getVarByName(f'f_tilde[{i}]').X)
            deltas.append(round(model.getVarByName(f'delta[{i}]').X))
        return u, deltas
    
def quarter_car(ss: SS, Np:int, dt: float, x: npt.NDArray, wdot: npt.NDArray):
    #   state
    #       1 zs-zu
    #       2 zu - zr
    #       3 zs'
    #       4 zu'
    #   input
    #       1 zr'
    #       2 f

    uf, deltasFront = solve(ss.par.csf, ss.par.csmin, ss.par.csmax, Np, x, wdot, ss.A, ss.B, ss.C, ss.D, ss.Q, ss.R)
    return uf[0], deltasFront[0]


# def state_mapping(x: npt.NDArray):
#     #   state
#     #       1 zs-zu
#     #       2 zu - zr
#     #       3 zs'
#     #       4 zu'
#     #   input
#     #       1 zr'
#     #       2 f
#     xf = np.array([[x[0, 0]], [x[4, 0]], [x[1, 0]], [x[5, 0]]])
#     xr = np.array([[x[2, 0]], [x[6, 0]], [x[3, 0]], [x[7, 0]]])
#     return xf, xr

# def state_setting(xf: npt.NDArray, xr: npt.NDArray):
#     return np.array([[xf[0, 0]], [xf[2, 0]], [xr[0, 0]], [xr[2, 0]], [xf[1, 0]], [xf[3, 0]], [xr[1, 0]], [xr[3, 0]]])