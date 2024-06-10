from gurobipy import Model, GRB, MVar
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters

par = Parameters(960, 1222, 40, 45, 200000,
                 200000, 18000, 22000, 1000,
                 1000, 1.3, 1.5)

def solve(Np, x: npt.NDArray, w: npt.NDArray, A: npt.NDArray, B: npt.NDArray, Q: npt.NDArray, R: npt.NDArray):
    # Params
    sigma = 6500 # [N]
    zmin = -1000
    zmax = 1000

    # Number of states (n) and inputs (m) and constraints (ncon)
    n = A.shape[0]
    m = B.shape[1]

    A_tilde = np.vstack([np.linalg.matrix_power(A, i+1) for i in range(Np)])
    print(A_tilde.shape)
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
    
    f_tilde = dict()
    w_tilde = dict()
    for i in range(Np):
        w_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'w_tilde[{i}]', lb=-1e-8, ub=1e-8)
        f_tilde[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f'f_tilde[{i}]', lb=-5000, ub=5000)

    u_list = []
    for i in range(Np):
        u_list.append([w_tilde[i]])
        u_list.append([f_tilde[i]])

    u_tilde = MVar.fromlist(u_list)

    model.update()

    obj = u_tilde.T @ H @ u_tilde + f @ u_tilde
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    model.write('model.lp')
    model.optimize()

    if model.status == GRB.OPTIMAL:
        xk = []
        for i in range(Np):
            xk.append(model.getVarByName(f'w_tilde[{i}]').X)
            xk.append(model.getVarByName(f'f_tilde[{i}]').X)
        return xk

A = np.array([
    [0, 1, 0, 0],
    [-par.ktf/par.muf, -par.csf/par.muf, par.ksf/par.muf, par.csf/par.muf],
    [0, -1, 0, 1],
    [0, par.csf/par.ms, -par.ksf/par.ms, -par.csf/par.ms]
])


print(np.linalg.eigvals(A))
print(np.linalg.eigvals(np.linalg.matrix_power(A, 4)))
B = np.array([
    [0, 0],
    [par.ktf/par.muf, par.ms/par.muf],
    [0, 0],
    [0, -1]
])

ss = signal.cont2discrete((A, B, np.eye(4), np.zeros((4, 2))), 1)
A = ss[0]
print(np.linalg.eigvals(A))
print(np.linalg.eigvals(np.linalg.matrix_power(A, 4)))
B = ss[1]

Q = np.zeros((4, 4))
Q[3, 3] = 1e-16
R = np.zeros((2, 2))
R[1, 1] = 1e-21
x = np.array([[0.01], [0], [0], [0]])
x1result = []
x2result = []
x3result = []
x4result = []
u1result = []
u2result = []
for _ in range(1):
    u = solve(10, x, np.zeros(1), A, B, Q, R)
    print(u)
    uk = [[u[0]], [u[1]]]
    x = A@x + B@uk
    x1result.append(x[0])
    x2result.append(x[1])
    x3result.append(x[2])
    x4result.append(x[3])
    u1result.append(uk[0])
    u2result.append(uk[1])

plt.plot(x1result)
plt.plot(x2result)
plt.plot(u1result)
plt.plot(u2result)
plt.grid()
plt.show()

# A = [[1, 0, -vr*sin(xrk(3))*k], [0, 1, vr*cos(xrk(3))*k], [0, 0, 1]]
# B = [cos(xrk(3))*k 0; sin(xrk(3))*k 0; 0 k];