import numpy as np
import pickle as pkl
from scipy.signal import cont2discrete

from roadsurface import isolatedBump, isolatedTable, isoRoad, isolatedCircle
from state_space_half_car import half_car_state_space, Parameters
from half import half_car
from quarter import quarter_car, state_mapping, state_setting
from metrics import wrms

par = Parameters(960, 1222, 40, 45, 200000,
                 200000, 18000, 22000, 1000,
                 1000, 1000/1.5, 1000*1.5, 1.3, 1.5)

par = Parameters(630, 1222, 37.5, 37.5, 210000, 210000, 29500, 29500, 1500, 1500, 1000, 4000, 1.3, 1.5)

# List of states:
# 1 - suspension deflection of the front car body
# 2 - vertical velocity of the front car body
# 3 - suspension deflection of the rear car body
# 4 - vertical velocity of the rear car body
# 5 - tire deflection of the front car body
# 6 - vertical velocity of the front wheel
# 7 - tire deflection of the rear car body
# 8 - vertical velocity of the rear wheel
state_half = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
state_quarter = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
state_pass = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

# Time
f = 30  # Hz
dt = 1/f  # s
endTime = 5  # s
tValues = np.arange(0, endTime, dt)  # the time array [s]
Np = 10  # length of the prediction horizon in points
Npfile = Np # file naming only, as Np is overwritten 

# Bump parameters (dependent on bump profile)
A = 0.1  # amplitude of the bump [m]
L = 0.5  # length of the bump [m]
V = 100 / 3.6  # velocity of the car [m/s]
tl = 0.1  # time of the bump [s]

l = tl * V  # position of the bump [m]
road_type = "iso"
k = 3
delay_samples = int((par.l1 + par.l2) / V * f)
# Front bump
match road_type:
    case "bump":
        road_profile_front = np.array(isolatedBump(f, A, V, l, L, endTime))
    case "table":
        road_profile_front = np.array(isolatedTable(f, V, l, endTime))
    case "iso":
        road_profile_front = np.array(isoRoad(f, V, k, endTime))
    case "circle":
        road_profile_front = np.array(isolatedCircle(f, V, endTime))

# Rear bump
road_profile_rear = np.zeros(len(tValues))
for i in range(len(road_profile_rear)):
    if i < delay_samples:
        road_profile_rear[i] = 0
    else:
        road_profile_rear[i] = road_profile_front[i - delay_samples]

# State Space
Ah, Bh, Fh, Ch, Dh = half_car_state_space(par)
Bh = np.concatenate((Bh, Fh), axis=1)
ssh = cont2discrete((Ah, Bh, Ch, Dh), dt)
A = ssh[0]
B = ssh[1]
C = ssh[2]
D = ssh[3]

# Simulation
n = len(tValues)  # number of samples
state_history = np.zeros((n, 8, 1))  # state history
state_quarter_history = np.zeros((n, 8, 1))  # quarter state history
state_pass_history = np.zeros((n, 8, 1))  # passive state history
output_history = np.zeros((n, 4, 1))  # acceleration history
output_quarter_history = np.zeros((n, 4, 1))  # quarter acceleration history
output_pass_history = np.zeros((n, 4, 1))  # passive acceleration history
u_history = np.zeros((n, 2, 1))  # control input history
u_quarter_history = np.zeros((n, 2, 1))  # quarter control input history

# generate road profile derivatives
road_profile_derivative_front = np.gradient(road_profile_front, dt)
road_profile_derivative_rear = np.gradient(road_profile_rear, dt)

# optimization variables
delta_front = []
delta_rear = []

for i in range(n):
    if Np > n - i:
        Np = n - i

    # create the road profile for the prediction horizon
    prediction_road_profile = np.zeros((Np, 2))
    for j in range(Np):
        prediction_road_profile[j] = np.array(
                [road_profile_derivative_front[i + j], road_profile_derivative_rear[i + j]])

    road_profile = np.array([[road_profile_derivative_front[i]], [road_profile_derivative_rear[i]]])

    state_history[i] = state_half
    state_quarter_history[i] = state_quarter
    state_pass_history[i] = state_pass

    mpc = half_car(par, Np, dt, state_half, prediction_road_profile[:, 0], prediction_road_profile[:, 1])
    force = np.array([[mpc[0]], [mpc[1]]])

    mpc2 = quarter_car(par, Np, dt, state_quarter, prediction_road_profile[:, 0], prediction_road_profile[:, 1])
    force2 = np.array([[mpc2[0]], [mpc2[1]]])

    u = np.array([[road_profile[0, 0]], [road_profile[1, 0]], [force[0, 0]], [force[1, 0]]])

    u_quarter = np.array([[road_profile[0, 0]], [road_profile[1, 0]], [force2[0, 0]], [force2[1, 0]]])

    upass = np.array([[road_profile[0, 0]], [road_profile[1, 0]], [0], [0]])

    next_state = A @ state_half + B @ u
    next_state_quarter = A @ state_quarter + B @ u_quarter
    next_state_passive = A @ state_pass + B @ upass
    output = C @ state_half + D @ u
    output_quarter = C @ state_quarter + D @ u_quarter
    output_passive = C @ state_pass + D @ upass

    state_half = next_state
    state_quarter = next_state_quarter
    state_pass = next_state_passive



    # save the state
    delta_front.append(mpc[2])
    delta_rear.append(mpc[3])

    output_history[i] = np.array([[output[0, 0]], [output[1, 0]], [0], [0]])
    output_quarter_history[i] = np.array([[output_quarter[0, 0]], [output_quarter[1, 0]], [0], [0]])
    output_pass_history[i] = np.array([[output_passive[0, 0]], [output_passive[1, 0]], [0], [0]])
    u_history[i] = np.array([[force[0, 0]], [force[1, 0]]])
    u_quarter_history[i] = np.array([[force2[0, 0]], [force2[1, 0]]])
    print(f"Step {i} of {n}")

damping_force_history = par.csf * (state_history[:, 1] - state_history[:, 5]) + u_history[:, 0]
damping_force_quarter = par.csf * (state_quarter_history[:, 1] - state_quarter_history[:, 5]) + u_quarter_history[:, 0]
damping_force_passive = par.csf * (state_pass_history[:, 1] - state_pass_history[:, 5])
deflection_velocity = state_history[:, 1] - state_history[:, 5]
deflection_velocity_passive = state_pass_history[:, 1] - state_pass_history[:, 5]
deflection_velocity_quarter = state_quarter_history[:, 1] - state_quarter_history[:, 5]

print(f"Active wrms: {wrms([], output_history[:, 0])}")
print(f"Quarter wrms: {wrms([], output_quarter_history[:, 0])}")
print(f"Passive wrms: {wrms([], output_pass_history[:, 0])}")

# save the results
results = [state_history, output_history, u_history, road_profile_front, road_profile_rear,
           damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history,
           output_pass_history, par.csf, par.csr, par.csmin, par.csmax, par, 
           state_quarter_history, output_quarter_history, u_quarter_history, damping_force_quarter, deflection_velocity_quarter]

# create a name for the file, based variables like endTime, f, tl, NP etc.
name = f"results_type_{road_type}_endT_{endTime}_f_{f}_tl_{tl}_Np_{Npfile}_quarter.pkl"

with open('results/' + name, 'wb') as f:
    pkl.dump(results, f)
