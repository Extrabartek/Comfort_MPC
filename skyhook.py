import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters
#from quarter import quarter_car
from control import lqr, forced_response, StateSpace
from roadsurface import isolatedBump, isolatedTable, isoRoad, isolatedBump

FROM_SLIDES = False

m_s = 960/2
m_u = 40
k_t = 200000
k_s = 18000
wn_s = np.sqrt(k_s*k_t / (k_s + k_t)/m_s)
d_s = 1000

## Quarter car from slides
if FROM_SLIDES:
    m_s = 410;                                           # sprung mass, kg
    m_u = 45;                                            # unsprung mass, kg
    k_t = 230000;                                        # tire vertical stiffness, N/m
    k_s = 25500;                                         # suspension vertical stiffness, N/m
    wn_s = np.sqrt(k_s*k_t / (k_s + k_t)/m_s);           # sprung mass natural frequency
    d_s = 0.3 * (2 * m_s * wn_s);                        # damping ratio

## Quarter car from paper - front

else:
    m_s = 960
    m_u = 40
    k_t = 200000
    k_s = 18000
    wn_s = np.sqrt(k_s*k_t / (k_s + k_t)/m_s)
    d_s = 1000

# print parametric values
print(f"m_s = {m_s}")
print(f"m_u = {m_u}")
print(f"k_t = {k_t}")
print(f"k_s = {k_s}")
print(f"wn_s = {wn_s}")
print(f"d_s = {d_s}")


# frequency range determination
n_points = 5000
w = np.linspace(0.1,100,n_points)*2*np.pi
# passive QC
# state matrix
A = np.array([[0, 1, 0, 0],                          # unsprung displacement - road disturbance
    [-k_t / m_u, -d_s / m_u, k_s / m_u, d_s / m_u],   # unsprung mass acceleration
    [0, -1, 0, 1],                                   # sprung displacement - unsprung displacement
    [0, d_s / m_s, -k_s / m_s, -d_s / m_s]])         # sprung mass acceleration
B = np.array([0, m_s / m_u, 0, -1])    # control matrix

B = np.reshape(B, (4, 1))

G = np.array([0, k_t / m_u, 0, 0])
# output matrix
C = np.array([[k_t, 0, 0, 0],       # tire force
    [0, 0, 1, 0],                   # suspension stroke
    A[3, :]])                       # assume A is a 2D array or matrix and we are selecting the 4th row]


Dw = np.array([[-k_t], [0], [0]])
Du = np.array([[0], [0], [-1]])                                    # feedthrough matrix
# Weights selection for use in performance index
r1 = 6e3;      # comfort weight
r2 = 5e3;      # road holding weight
r3 = 5e5;         # control effort weight
Rxx = (A[3,:].T) @ A[3,:] + np.diag([r1, 0, r2, 0])
Rxx[3,3] = 5e7
Rxu = np.reshape(-A[3,:].T, (4, 1))
Ruu = np.reshape(np.array([1 + r3]), (1, 1))    

K, S, E = lqr(A, B, Rxx, Ruu, Rxu)
print(K)

A_cl = A - B @ K
C_cl = (C - Du @ K)
# Define the initial state
x0 = np.array([0, 0, 0, 0])

# Define the time vector
t = np.linspace(0, 10, 10000)

# Define the disturbance input as a time series
# Example: a sinusoidal disturbance
disturbance_sin = np.exp(-1*t) * 0.1*np.sin(2*np.pi * t)  # replace with your disturbance time series
disturbance = np.zeros_like(disturbance_sin)
disturbance[99:199] = disturbance_sin[:100]
disturbance = disturbance.reshape(-1, 1).T  # ensure it has the correct shape

## Time init
f_road = 1000           # Hz
## Tunable parameters (dependent on bump surface)
A_road = 0.1            # mS
V_road = 25 / 3.6       # km/h
l_road = 10             # m
L_road = 0.5             # m

## Generate profile
profileBump = isolatedBump(f_road, A_road, V_road, l_road, L_road, 10)
num_samples = len(profileBump)-1
z_r = profileBump[:-1]
print(C_cl[0,:])

# Simulate the response of the closed-loop system to the disturbance
#sys_cl = StateSpace(A_cl, G, np.array([1,0,0,0]), Dw[0])
G = G.reshape(4, 1)
sys_cl = StateSpace(A_cl, np.hstack((B, G)), C, np.hstack((Dw, np.zeros((C.shape[0], G.shape[1])))))
sys_ol = StateSpace(A, np.hstack((B, G)), C, np.hstack((Dw, np.zeros((C.shape[0], G.shape[1])))))    
u = np.vstack((np.zeros((1, num_samples)), z_r))

t, y, x = forced_response(sys_ol, T=t, U=u, X0=x0, return_x=True)

print(f"y shape: {y.shape}")

x_dot = np.gradient(x, t, axis=1)

u = -(K @ x).T

PLOT = 1

if PLOT == 1:
    # Plot disturbance input
    plt.figure(figsize=(10, 4))
    plt.plot(t, profileBump[:-1], label='Disturbance')
    plt.xlabel('Time (s)')
    plt.ylabel('Disturbance')
    plt.title('Disturbance Input')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the results
    plt.figure(figsize=(10, 8))
    #plt.plot(t, x[0,:], label='Z_u deflection')
    #plt.plot(t, x[1,:], label='Z_u velocity')
    plt.plot(t, x[2,:], label='Z_s deflection')
    plt.plot(t, x[3,:], label='Z_s velocity')
    plt.xlim([0, 10])
    #plt.ylim([-1,1])
    plt.xlabel('Time (s)')
    plt.ylabel('State values')
    plt.title('System Response with LQR Controller and Disturbance')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.plot(t, x_dot[1,:], label='Z_u acceleration')
    plt.plot(t, x_dot[3,:], label='Z_s acceleration')
    plt.plot(t, u, label='Control input')
    plt.xlim([0, 10])
    plt.ylim([-50,50])
    plt.xlabel('Time (s)')
    plt.ylabel('State values')
    plt.title('System Response with LQR Controller and Disturbance')
    plt.legend()
    plt.grid()
    plt.show()
print(f"Max positive a_u: {np.max(x_dot[1,:])}")
print(f"Max positive a_s: {np.max(x_dot[3,:])}")
print(f"Max negative a_u: {np.min(x_dot[1,:])}")
print(f"Max negative a_s: {np.min(x_dot[3,:])}")
print(f"Max unsprung mass deflection: {np.max(x[0,:])}")
print(f"Max sprung mass deflection: {np.max(x[2,:])}")
print(f"Max control input: {np.max(u)}")
print(f"Max negative control input: {np.min(u)}")