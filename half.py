from gurobipy import Model, GRB, MVar
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters, half_car_state_space

def solve(par: Parameters, Np, x: npt.NDArray, wf: npt.NDArray, wr: npt.NDArray, A: npt.NDArray, B: npt.NDArray, C: npt.NDArray, D: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
    # Params
    eps = 1e-8
    M = 100000
    # sigma = 6500 # [N]

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
    H = B_c_tilde.T @ Q_tilde @ B_c_tilde + R_tilde # checked
    f = 2 * x.T @ A_c_tilde.T @ Q_tilde @ B_c_tilde # checked

    model = Model('MPC controller')
    #model.Params.LogToConsole = 0
    
    wf_tilde = dict()
    ff_tilde = dict()
    wr_tilde = dict()
    fr_tilde = dict()
    wf = np.reshape(wf, (Np, 1))
    wr = np.reshape(wr, (Np, 1))
    for i in range(Np):
        wf_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'wf_tilde[{i}]', lb=wf[i, 0]-eps, ub=wf[i, 0]+eps)
        wr_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'wb_tilde[{i}]', lb=wr[i, 0]-eps, ub=wr[i, 0]+eps)
        ff_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'ff_tilde[{i}]', lb=-GRB.INFINITY, ub=GRB.INFINITY)
        fr_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'fb_tilde[{i}]', lb=-GRB.INFINITY, ub=GRB.INFINITY)

    u_list = []
    for i in range(Np):
        u_list.append([wf_tilde[i]])
        u_list.append([wr_tilde[i]])
        u_list.append([ff_tilde[i]])
        u_list.append([fr_tilde[i]])

    u_tilde = MVar.fromlist(u_list)

    deltaf = dict()
    deltar = dict()
    z1 = dict()
    z2 = dict()
    z3 = dict()
    z4 = dict()
    for i in range(Np):
        deltaf[i] = model.addVar(vtype=GRB.BINARY, name='delta_f[{}]'.format(i))
        deltar[i] = model.addVar(vtype=GRB.BINARY, name='delta_r[{}]'.format(i))
        z1[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z1[{}]'.format(i))
        z2[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z2[{}]'.format(i))
        z3[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z3[{}]'.format(i))
        z4[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z4[{}]'.format(i))

    model.update()
    for i in range(Np):
        # Front -> map to binary
        model.addConstr(A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 5, :] @ x + B_tilde[i*n + 5, 0:i*m+m] @ u_tilde[0:i*m+m] >= -M*(1-deltaf[i]))
        model.addConstr(A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 5, :] @ x + B_tilde[i*n + 5, 0:i*m+m] @ u_tilde[0:i*m+m] <= M*deltaf[i])
        # Rear -> map to binary
        model.addConstr(A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 7, :] @ x + B_tilde[i*n + 7, 0:i*m+m] @ u_tilde[0:i*m+m] >= -M*(1-deltar[i]))
        model.addConstr(A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                      - A_tilde[i*n + 7, :] @ x + B_tilde[i*n + 7, 0:i*m+m] @ u_tilde[0:i*m+m] <= M*deltar[i])
        
        # Front
        model.addConstr(z1[i] <= eps + M*deltaf[i])
        model.addConstr(z1[i] >= -M*deltaf[i])
        model.addConstr(z1[i] <= eps + u_tilde[i*m + 2, 0] + (par.csf - par.csmin) * (A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                          - A_tilde[i*n + 5, :] @ x + B_tilde[i*n + 5, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                          + M*(1 - deltaf[i]))
        model.addConstr(z1[i] >= u_tilde[i*m + 2, 0] + (par.csf - par.csmin) * (A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                    - A_tilde[i*n + 5, :] @ x + B_tilde[i*n + 5, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                    - M*(1 - deltaf[i]))
        
        model.addConstr(z2[i] <= eps + M*deltaf[i])
        model.addConstr(z2[i] >= -M*deltaf[i])
        model.addConstr(z2[i] <= eps + u_tilde[i*m + 2, 0] + (par.csf - par.csmax) * (A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                          - A_tilde[i*n + 5, :] @ x + B_tilde[i*n + 5, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                          + M*(1 - deltaf[i]))
        model.addConstr(z2[i] >= u_tilde[i*m + 2, 0] + (par.csf - par.csmax) * (A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                    - A_tilde[i*n + 5, :] @ x + B_tilde[i*n + 5, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                    - M*(1 - deltaf[i]))

        model.addConstr(-u_tilde[i*m + 2, 0] - (par.csf - par.csmin) * (A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                            - A_tilde[i*n + 5, :] @ x + B_tilde[i*n + 5, 0:i*m+m] @ u_tilde[0:i*m+m])
                                                            + 2*z1[i] >= 0)
        model.addConstr(-u_tilde[i*m + 2, 0] - (par.csf - par.csmax) * (A_tilde[i*n + 1, :] @ x + B_tilde[i*n + 1, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                            - A_tilde[i*n + 5, :] @ x + B_tilde[i*n + 5, 0:i*m+m] @ u_tilde[0:i*m+m])
                                                            + 2*z2[i] <= 0)
        
        # Rear
        model.addConstr(z3[i] <= eps + M*deltar[i])
        model.addConstr(z3[i] >= -M*deltar[i])
        model.addConstr(z3[i] <= eps + u_tilde[i*m + 3, 0] + (par.csr - par.csmin) * (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                          - A_tilde[i*n + 7, :] @ x + B_tilde[i*n + 7, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                          + M*(1 - deltar[i]))
        model.addConstr(z3[i] >= u_tilde[i*m + 3, 0] + (par.csr - par.csmin) * (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                    - A_tilde[i*n + 7, :] @ x + B_tilde[i*n + 7, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                    - M*(1 - deltar[i]))
        
        model.addConstr(z4[i] <= eps + M*deltar[i])
        model.addConstr(z4[i] >= -M*deltar[i])
        model.addConstr(z4[i] <= eps + u_tilde[i*m + 3, 0] + (par.csr - par.csmax) * (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                          - A_tilde[i*n + 7, :] @ x + B_tilde[i*n + 7, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                          + M*(1 - deltar[i]))
        model.addConstr(z4[i] >= u_tilde[i*m + 3, 0] + (par.csr - par.csmax) * (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                                    - A_tilde[i*n + 7, :] @ x + B_tilde[i*n + 7, 0:i*m+m] @ u_tilde[0:i*m+m]) 
                                                                    - M*(1 - deltar[i]))

        model.addConstr(-u_tilde[i*m + 3, 0] - (par.csr - par.csmin) * (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                            - A_tilde[i*n + 7, :] @ x + B_tilde[i*n + 7, 0:i*m+m] @ u_tilde[0:i*m+m])
                                                            + 2*z3[i] >= 0)
        model.addConstr(-u_tilde[i*m + 3, 0] - (par.csr - par.csmax) * (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]
                                                            - A_tilde[i*n + 7, :] @ x + B_tilde[i*n + 7, 0:i*m+m] @ u_tilde[0:i*m+m])
                                                            + 2*z4[i] <= 0)
        
        
    model.update()
    obj = u_tilde.T @ H @ u_tilde + f @ u_tilde
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    # model.write('model.lp')
    model.optimize()

    if model.status == GRB.OPTIMAL:
        u_f = []
        u_r = []
        deltas_f = []
        deltas_r = []
        for i in range(Np):
            u_f.append(model.getVarByName(f'ff_tilde[{i}]').X)
            u_r.append(model.getVarByName(f'fb_tilde[{i}]').X)
            deltas_f.append(model.getVarByName(f'delta_f[{i}]').X)
            deltas_r.append(model.getVarByName(f'delta_r[{i}]').X)
        return u_f, u_r, deltas_f, deltas_r 
    
def half_car(par: Parameters, Np: int, dt: float, x:npt.NDArray, wfdot: npt.NDArray, wbdot: npt.NDArray):
    # State space matrices
    A, B, F, C, D = half_car_state_space(par)
    B = np.concatenate((B, F), axis=1)

    Q = np.array([[1, 0],
                  [0, np.sqrt(par.l1*par.l2)]])
    
    R = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    ss = signal.cont2discrete((A, B, C, D), dt)

    A = ss[0]
    B = ss[1]
    C = ss[2]
    D = ss[3]

    uf, ub, df, dr = solve(par, Np, x, wfdot, wbdot, A, B, C, D, Q, R)
    return uf[0], ub[0], df[0], dr[0]

    
if __name__ == '__main__':
    half_car(par = Parameters(960, 1222, 40, 45, 200000,
                    200000, 18000, 22000, 1000,
                    1000, 1000/1.5, 1000*1.5, 1.3, 1.5), 
             Np=10, 
             dt=0.01, 
             x=np.array([[0], [0.1], [0], [0.1], [0], [0.1], [0], [0.1]]), 
             wfdot=np.zeros((10, 1)),
             wbdot=np.zeros((10, 1))
            )
