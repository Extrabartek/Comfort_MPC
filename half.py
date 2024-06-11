from gurobipy import Model, GRB, MVar
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters, half_car_state_space

def solve(Np, x: npt.NDArray, wf: npt.NDArray, wb: npt.NDArray, A: npt.NDArray, B: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
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
    
    wf_tilde = dict()
    ff_tilde = dict()
    wb_tilde = dict()
    fb_tilde = dict()
    for i in range(Np):
        wf_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'wf_tilde[{i}]', lb=wf[i, 0]-1e-8, ub=wf[i, 0]+1e-8)
        wb_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'wb_tilde[{i}]', lb=wb[i, 0]-1e-8, ub=wb[i, 0]+1e-8)
        ff_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'ff_tilde[{i}]', lb=-GRB.INFINITY, ub=GRB.INFINITY)
        fb_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'fb_tilde[{i}]', lb=-GRB.INFINITY, ub=GRB.INFINITY)

    u_list = []
    for i in range(Np):
        u_list.append([wf_tilde[i]])
        u_list.append([wb_tilde[i]])
        u_list.append([ff_tilde[i]])
        u_list.append([fb_tilde[i]])

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
        uf = []
        ub = []
        for i in range(Np):
            uf.append(model.getVarByName(f'ff_tilde[{i}]').X)
            ub.append(model.getVarByName(f'fb_tilde[{i}]').X)
        return uf, ub
    
def half_car(par: Parameters, Np: int, dt: float, x:npt.NDArray, wfdot: npt.NDArray, wbdot: npt.NDArray):
    # State space matrices
    A, B, F, C, E = half_car_state_space(par)

    B = np.concatenate((B, F))

    ss = signal.cont2discrete((A, B, C, E), dt)

    A = ss[0]
    B = ss[1]

    uf, ub = solve(Np, x, wfdot, wbdot, A, B, np.eye(8), np.eye(4))
    return uf[0], ub[0]

    
if __name__ == '__main__':
    half_car(par = Parameters(960, 1222, 40, 45, 200000,
                    200000, 18000, 22000, 1000,
                    1000, 1.3, 1.5), 
             Np=10, 
             dt=0.01, 
             x=np.array([[0], [0.1], [0], [0.1], [0], [0.1], [0], [0.1]]), 
             wfdot=np.zeros((10, 1)),
             wbdot=np.zeros((10, 1))
            )
