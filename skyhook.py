import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters
#from quarter import quarter_car
from control import lqr, forced_response, StateSpace

FROM_SLIDES = False

m_s = 960
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
r1 = 4e+4;      # comfort weight
r2 = 5e+3;      # road holding weight
r3 = 0;         # control effort weight
Rxx = (A[3,:].T) @ A[3,:] + np.diag([r1, 0, r2, 0])
Rxu = np.reshape(-A[3,:].T, (4, 1))
Ruu = np.reshape(np.array([1 + r3]), (1, 1))    

K, S, E = lqr(A, B, Rxx, Ruu, Rxu)

print(K)

A_cl = A - B @ K
C_cl = (C - Du @ K)
# Define the initial state
x0 = np.array([0, 0, 0, 0])

# Define the time vector
t = np.linspace(0, 10, 1000)

# Define the disturbance input as a time series
# Example: a sinusoidal disturbance
disturbance_sin = np.exp(-10*t) * 5*np.sin(100*2*np.pi * t)  # replace with your disturbance time series
disturbance = np.zeros_like(disturbance_sin)
disturbance[99:199] = disturbance_sin[:100]
disturbance = disturbance.reshape(-1, 1).T  # ensure it has the correct shape

# Simulate the response of the closed-loop system to the disturbance
sys_cl = StateSpace(A_cl, G, C_cl[0,:], Dw[0])


t, y, x = forced_response(sys_cl, T=t, U=disturbance, X0=x0, return_x=True)

x_dot = np.gradient(x, t, axis=1)

# Plot disturbance input
plt.figure(figsize=(10, 4))
plt.plot(t, disturbance[0, :], label='Disturbance')
plt.xlabel('Time (s)')
plt.ylabel('Disturbance')
plt.title('Disturbance Input')
plt.legend()
plt.grid()
plt.show()

# Plot the results
plt.figure(figsize=(10, 8))
# plt.plot(t, x[0,:], label='Z_s deflection')
# plt.plot(t, x[1,:], label='Z_s velocity')
# plt.plot(t, x[2,:], label='Z_u deflection')
# plt.plot(t, x[3,:], label='Z_u velocity')
plt.plot(t, x_dot[1,:], label='Z_s acceleration')
plt.plot(t, x_dot[3,:], label='Z_u acceleration')
plt.xlim([0.8, 4])
plt.ylim([-10, 20])
plt.xlabel('Time (s)')
plt.ylabel('State values')
plt.title('System Response with LQR Controller and Disturbance')
plt.legend()
plt.grid()
plt.show()