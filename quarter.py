from gurobipy import Model, GRB, MVar
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters

def solve(cs: float, Np, x: npt.NDArray, w: npt.NDArray, A: npt.NDArray, B: npt.NDArray, C: npt.NDArray, D: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
    # Params
    eps = 1e-8
    M = 100000
    sigma = 6500 # [N]
    # kappa = 7000 # [Ns/m]
    # kappa = cs # [Ns/m]
    cmin = cs/1.5
    cmax = cs*1.5

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
    z = dict()
    z1 = dict()
    z2 = dict()
    for i in range(Np):
        delta[i] = model.addVar(vtype=GRB.BINARY, name='delta[{}]'.format(i))
        # z[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z[{}]'.format(i))
        z1[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z1[{}]'.format(i))
        z2[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z2[{}]'.format(i))

    model.update()
    for i in range(Np):
        pass
        # model.addConstr(u_tilde[i*2 + 1, 0] <= sigma)
        # model.addConstr(u_tilde[i*2 + 1, 0] >= -sigma)
        # model.addConstr(A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #               - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m] >= -M * (1 - delta[i]))
        # model.addConstr(A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #               - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m] <= M * delta[i])
        # model.addGenConstrIndicator(delta[i], 1, u_tilde[i*2 + 1, 0], GRB.GREATER_EQUAL, eps)
        # model.addGenConstrIndicator(delta[i], 0, u_tilde[i*2 + 1, 0], GRB.LESS_EQUAL, 0)
        # model.addConstr(z[i] <= eps + M * delta[i])
        # model.addConstr(z[i] >= - M * delta[i])
        # model.addConstr(z[i] <= eps + u_tilde[i*2 + 1, 0] - kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                     - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                     + M * (1 - delta[i]))
        # model.addConstr(z[i] >= u_tilde[i*2 + 1, 0] - kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                     - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                     - M * (1 - delta[i]))
        # model.addConstr(u_tilde[i*2 + 1, 0] - kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                              - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                              - 2*z[i] >= 0)
        # model.addConstr(- u_tilde[i*2 + 1, 0] - kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                              - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                              - 2*delta[i]*(- u_tilde[i*2 + 1, 0] - kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                      - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])) >= 0)
        
        
        # model.addConstr((A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) <= M*delta[i])
        # model.addConstr((A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) >= -M*(1-delta[i]))
        #
        #
        #
        # model.addConstr(z1[i] <= eps + M*delta[i])
        # model.addConstr(z1[i] >= -M*delta[i])
        # model.addConstr(z1[i] <= eps + u_tilde[i*2 + 1, 0] + (cs - cmin) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                                   - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                                   + M*(1 - delta[i]))
        # model.addConstr(z1[i] >= u_tilde[i*2 + 1, 0] + (cs - cmin) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                             - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                             - M*(1 - delta[i]))
        # model.addConstr(z2[i] <= eps + M*delta[i])
        # model.addConstr(z2[i] >= -M*delta[i])
        # model.addConstr(z2[i] <= eps + u_tilde[i*2 + 1, 0] + (cs - cmax) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                                   - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                                   + M*(1 - delta[i]))
        # model.addConstr(z2[i] >= u_tilde[i*2 + 1, 0] + (cs - cmax) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                             - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                             - M*(1 - delta[i]))
        #
        # model.addConstr(-u_tilde[i*2 + 1, 0] - (cs - cmin) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                     - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                     + 2*z1[i] >= 0)
        # model.addConstr(-u_tilde[i*2 + 1, 0] - (cs - cmax) * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                     - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                     + 2*z2[i] <= 0)

        
        # model.addConstr(z[i] <= eps + M * delta[i])
        # model.addConstr(z[i] >= - M * delta[i])
        # model.addConstr(z[i] <= eps + u_tilde[i*2 + 1, 0] + kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                     - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                     + M * (1 - delta[i]))
        # model.addConstr(z[i] >= u_tilde[i*2 + 1, 0] + kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                     - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                     - M * (1 - delta[i]))
        
        # model.addConstr(-u_tilde[i*2 + 1, 0] - kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                               - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) + 2*z[i] >= 0)




        # model.addConstr(z1[i] <= eps + M*delta[i])
        # model.addConstr(z1[i] >= -M*delta[i])
        # model.addConstr(z1[i] <= eps + (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]) + M*(1 - delta[i]))
        # model.addConstr(z1[i] >= (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]) - M*(1 - delta[i]))
        # model.addConstr(z2[i] <= eps + M*delta[i])
        # model.addConstr(z2[i] >= -M*delta[i])
        # model.addConstr(z2[i] <= eps + (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) + M*(1 - delta[i]))
        # model.addConstr(z2[i] >= (A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]) - M*(1 - delta[i]))
        # model.addConstr(u_tilde[i*2 + 1, 0] <= -kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                               - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                               + 2*kappa*(z1[i] -z2[i]))
        # model.addConstr(u_tilde[i*2 + 1, 0] >= kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                               - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                               - 2*kappa*(z1[i] -z2[i]))

        # model.addConstr(u_tilde[i*2 + 1, 0] <= - kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                 - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                              + 2*delta[i]*kappa *(A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                                 - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]))
        # model.addConstr(u_tilde[i*2 + 1, 0] >= kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                 - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                              - 2*delta[i]*kappa *(A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                                 - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]))

        # model.addConstr(u_tilde[i*2 + 1, 0] * u_tilde[i*2 + 1, 0] >= kappa * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                                     - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m])
        #                                                                    * (A_tilde[i*n + 2, :] @ x + B_tilde[i*n + 2, 0:i*m+m] @ u_tilde[0:i*m+m]
        #                                                                     - A_tilde[i*n + 3, :] @ x + B_tilde[i*n + 3, 0:i*m+m] @ u_tilde[0:i*m+m]))

    model.update()
    obj = u_tilde.T @ H @ u_tilde + f @ u_tilde
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    # model.write('model.lp')
    model.optimize()

    if model.status == GRB.OPTIMAL:
        u = []
        deltas = []
        z = []
        for i in range(Np):
            u.append(model.getVarByName(f'f_tilde[{i}]').X)
            deltas.append(round(model.getVarByName(f'delta[{i}]').X))
            # z.append(model.getVarByName(f'z[{i}]').X)
            z.append(0)
        return u, deltas, z
    
def quarter_car(par: Parameters, Np:int, dt: float, x: npt.NDArray, wfdot: npt.NDArray, wrdot: npt.NDArray):
    #   state
    #       1 zs-zu
    #       2 zu - zr
    #       3 zs'
    #       4 zu'
    #   input
    #       1 zr'
    #       2 f
    
    Af = np.array([
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [-par.ksf/(par.ms/2), 0, -par.csf/(par.ms/2), par.csf/(par.ms/2)],
        [par.ksf/par.muf, -par.ktf/par.muf, par.csf/par.muf, -par.csf/par.muf]
    ])

    Bf = np.array([
        [0, 0],
        [-1, 0],
        [0, -1/(par.ms/2)],
        [0, 1/par.muf]
    ])

    Cf = np.array([
        [-par.ksf/(par.ms/2), 0, -par.csf/(par.ms/2), par.csf/(par.ms/2)]])
    
    Df = np.array([[0, -1/(par.ms/2)]])

    Ar = np.array([
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [-par.ksr/(par.ms/2), 0, -par.csr/(par.ms/2), par.csr/(par.ms/2)],
        [par.ksr/par.mur, -par.ktr/par.mur, par.csr/par.mur, -par.csr/par.mur]
    ])

    Br = np.array([
        [0, 0],
        [-1, 0],
        [0, -1/(par.ms/2)],
        [0, 1/par.mur]
    ])

    Cr = np.array([
        [-par.ksr/(par.ms/2), 0, -par.csr/(par.ms/2), par.csr/(par.ms/2)]])
    
    Dr = np.array([[0, -1/(par.ms/2)]])

    Q = np.array([[1]])

    # R = np.array([[0, 0], [0, 1/20000]])
    R = np.array([[0, 0], [0, 0]])

    ssf = signal.cont2discrete((Af, Bf, Cf, Df), dt)
    ssr = signal.cont2discrete((Ar, Br, Cr, Dr), dt)

    Af = ssf[0]
    Bf = ssf[1]
    Cf = ssf[2]
    Df = ssf[3]
    Ar = ssr[0]
    Br = ssr[1]
    Cr = ssr[2]
    Dr = ssr[3]

    xf, xr = state_mapping(x)
    uf, deltasFront, zf = solve(par.csf, Np, xf, wfdot, Af, Bf, Cf, Df, Q, R)
    ub, deltasRear, zr = solve(par.csr, Np, xr, wrdot, Ar, Br, Cr, Dr, Q, R)
    return uf[0], ub[0], deltasFront[0], deltasRear[0], zf[0], zr[0]


def state_mapping(x: npt.NDArray):
    #   state
    #       1 zs-zu
    #       2 zu - zr
    #       3 zs'
    #       4 zu'
    #   input
    #       1 zr'
    #       2 f
    xf = np.array([[x[0, 0]], [x[4, 0]], [x[1, 0]], [x[5, 0]]])
    xr = np.array([[x[2, 0]], [x[6, 0]], [x[3, 0]], [x[7, 0]]])
    return xf, xr


if __name__ == "__main__":
    quarter_car(par = Parameters(960, 1222, 40, 45, 200000,
                    200000, 18000, 22000, 1000,
                    1000, 1.3, 1.5),
                Np=10, 
                dt=0.01, 
                x=np.array([[0], [0.1], [0], [0.1], [0], [0.1], [0], [0.1]]), 
                wfdot=np.zeros((10, 1)),
                wrdot=np.zeros((10, 1))
               )