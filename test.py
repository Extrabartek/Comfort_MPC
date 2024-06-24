import numpy as np
import pickle as pkl
import time

from roadsurface import isolatedBump, isolatedTable, isoRoad, isolatedCircle
from state_space_half_car import half_car_state_space, Parameters
from quarter import quarter_car, state_mapping
from metrics import wrms

par = Parameters(960, 1222, 40, 45, 200000,
                 200000, 18000, 22000, 1000,
                 1000, 1000/1.5, 1000*1.5, 1.3, 1.5)

par = Parameters(630, 1222, 37.5, 37.5, 210000, 210000,
                 29500, 29500, 1500, 1500, 1500/1.5, 1500*1.5, 1.3, 1.5)

# Define the state-space matrices
state = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
state_passive = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

# List of states:
# 1 - suspension deflection of the front car body
# 2 - vertical velocity of the front car body
# 3 - suspension deflection of the rear car body
# 4 - vertical velocity of the rear car body
# 5 - tire deflection of the front car body
# 6 - vertical velocity of the front wheel
# 7 - tire deflection of the rear car body
# 8 - vertical velocity of the rear wheel

# Time init
f = 1000  # Hz
endTime = 0.5  # s
tValues = np.arange(0, endTime, 1 / f)  # the time array [s]

# Tunable parameters (dependent on bump profile)
A = 0.1  # amplitude of the bump [m]
V = 25 / 3.6  # velocity of the car [m/s]
tl = 0.01  # time of the bump [s]
l = tl * V  # position of the bump [m]
L = 0.5  # length of the bump [m]
road_type = "bump"  # bump, table, iso, circle

match road_type:
    case "bump":
        road_profile_front = np.array(isolatedBump(f, A, V, l, L, endTime))
    case "table":
        road_profile_front = np.array(isolatedTable(f, V, l, endTime))
    case "iso":
        road_profile_front = np.array(isoRoad(f, V, endTime))
    case "circle":
        road_profile_front = np.array(isolatedCircle(f, V, endTime))


# run the script to create the road profile
# road_profile_front = np.array(isoRoad(f, V, endTime))
# road_profile_front = np.array(isolatedBump(f, A, V, l, L, endTime))

# Calculate the delay in samples
delay_samples = int((par.l1 + par.l2) / V * f)

# Initialize the rear road profile
road_profile_rear = np.zeros(len(tValues))

# Create the delayed rear road profile
for i in range(len(road_profile_rear)):
    if i < delay_samples:
        road_profile_rear[i] = 0
    else:
        road_profile_rear[i] = road_profile_front[i - delay_samples]

# The simulation loop

dt = 1 / f  # time step [s]
Np = 10  # length of the prediction horizon in points
t_prediction = 0.05  # length of the prediction horizon in seconds
dt_prediction = t_prediction / Np  # time step of the prediction horizon [s]
n = len(tValues)  # number of samples
state_history = np.zeros((n, 8, 1))  # state history
derivative_history = np.zeros((n, 4, 1))  # derivative history
acceleration_history = np.zeros((n, 1, 1))  # acceleration history
u_history = np.zeros((n, 2, 1))

passive_state = np.zeros((n, 8, 1))
passive_derivative = np.zeros((n, 4, 1))
passive_acceleration = np.zeros((n, 1, 1))
# generate road profile derivatives
road_profile_derivative_front = np.gradient(road_profile_front, dt)
road_profile_derivative_rear = np.gradient(road_profile_rear, dt)

delta_front = []
delta_rear = []

zlistf = []
zlistr = []

# get the state-spaces matrices
A = np.array([
    [0, 0, 1, -1],
    [0, 0, 0, 1],
    [-par.ksf / (par.ms / 2), 0, -par.csf / (par.ms / 2), par.csf / (par.ms / 2)],
    [par.ksf / par.muf, -par.ktf / par.muf, par.csf / par.muf, -par.csf / par.muf]
])

B = np.array([
    [0, 0],
    [-1, 0],
    [0, -1 / (par.ms / 2)],
    [0, 1 / par.muf]
])

C = np.array([
    [-par.ksf / (par.ms / 2), 0, -par.csf / (par.ms / 2), par.csf / (par.ms / 2)]])

D = np.array([[0, -1 / (par.ms / 2)]])

for i in range(n):
    # create the road profile based on the derivative
    road_profile = np.array([[road_profile_derivative_front[i]], [road_profile_derivative_rear[i]]])

    # create the road profile for the prediction horizon
    prediction_road_profile = np.zeros((Np, 2))

    for j in range(Np):
        if i + j >= len(tValues):
            prediction_road_profile[j] = np.array([0, 0])
        else:
            prediction_road_profile[j] = np.array(
                [road_profile_derivative_front[i + j], road_profile_derivative_rear[i + j]])

    # for j in range(Np):
    #     index = i + j * int(dt_prediction / dt)
    #     if index >= len(tValues):
    #         prediction_road_profile[j] = np.array([0, 0])
    #     else:
    #         prediction_road_profile[j] = np.array([road_profile_derivative_front[index], road_profile_derivative_rear[index]])

    # solve for the control input
    u = quarter_car(par, Np, dt, state, prediction_road_profile[:, 0], prediction_road_profile[:, 1])
    delta_front.append(u[2])
    delta_rear.append(u[3])
    zlistf.append(u[4])
    zlistr.append(u[5])
    u = np.array([[u[0]], [u[1]]])
    # u = np.array([[100], [0]])
    upassive = np.array([[0], [0]])
    # road_profile = np.array([[0], [0]])
    # calculate the derivativec
    xf, _ = state_mapping(state)
    xfpass, _ = state_mapping(state_passive)
    derivative = A @ xf + B @ np.array([[road_profile[0, 0]], [u[0, 0]]])

    derivative_passive = A @ xfpass + B @ np.array([[road_profile[0, 0]], [upassive[0, 0]]])

    # calculate the acceleration
    acceleration = C @ xf + D @ np.array([[road_profile[0, 0]], [u[0, 0]]])
    acceleration_passive = C @ xfpass + D @ np.array([[road_profile[0, 0]], [upassive[0, 0]]])

    # update the state
    change = dt * derivative
    state = state + np.array([[change[0, 0]], [change[2, 0]], [0], [0], [change[1, 0]], [change[3, 0]], [0], [0]])
    change_passive = dt * derivative_passive
    state_passive = state_passive + np.array(
        [[change_passive[0, 0]], [change_passive[2, 0]], [0], [0], [change_passive[1, 0]], [change_passive[3, 0]], [0],
         [0]])

    # save the state
    state_history[i] = state
    derivative_history[i] = derivative
    acceleration_history[i] = acceleration
    passive_state[i] = state_passive
    passive_acceleration[i] = acceleration_passive
    passive_derivative[i] = derivative_passive
    u_history[i] = u
    print(f"Step {i} of {n}")

# create seperated sub-figures for the acceleration, the state and the road profile
damping_force_history = par.csf * (state_history[:, 1] - state_history[:, 5]) + u_history[:, 0]
z_values = state_history[:, 1] - state_history[:, 5]
passive_damping_force = par.csf * (passive_state[:, 1] - passive_state[:, 5])
passive_z_values = passive_state[:, 1] - passive_state[:, 5]

print(wrms([], acceleration_history[:, 0]))
print(wrms([], passive_acceleration[:, 0]))

# save the results
results = [state_history, derivative_history, acceleration_history, u_history, road_profile_front, road_profile_rear,
           damping_force_history, z_values, passive_damping_force, passive_z_values, tValues, passive_state,
           passive_acceleration, road_profile_front, delta_front, delta_rear, zlistf]

# create a name for the file, based variables like endTime, f, tl, NP etc.
name = f"results_type_{road_type}_endT_{endTime}_f_{f}_tl_{tl}_Np_{Np}_no_constrains.pkl"

with open('results/' + name, 'wb') as f:
    pkl.dump(results, f)
