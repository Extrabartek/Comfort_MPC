import numpy as np
import matplotlib.pyplot as plt

from roadsurface import isolatedBump, isolatedTable, isoRoad
from state_space_half_car import half_car_state_space, Parameters
from quarter import quarter_car, state_mapping

par = Parameters(960, 1222, 40, 45, 200000,
                 200000, 18000, 22000, 1000,
                 1000, 1.3, 1.5)

par = Parameters(630, 1222, 37.5, 37.5, 210000, 210000,
                 29500, 29500, 1500, 1500, 1.3, 1.5)

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
endTime = 0.3  # s
tValues = np.arange(0, endTime, 1 / f)  # the time array [s]

# Tunable parameters (dependent on bump profile)
A = 0.1  # amplitude of the bump [m]
V = 25 / 3.6  # velocity of the car [m/s]
tl = 0.05  # time of the bump [s]
l = tl * V  # position of the bump [m]
L = 0.5  # length of the bump [m]

# run the script to create the road profile
# road_profile_front = np.array(isoRoad(f, V, endTime))
road_profile_front = np.array(isolatedBump(f, A, V, l, L, endTime))

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

# get the state-spaces matrices
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
    [-par.ksf/(par.ms/2), 0, -par.csf/(par.ms/2), par.csf/(par.ms/2)]])

D = np.array([[0, -1/(par.ms/2)]])

for i in range(n):
    # create the road profile based on the derivative
    road_profile = np.array([[road_profile_derivative_front[i]], [road_profile_derivative_rear[i]]])

    # create the road profile for the prediction horizon
    prediction_road_profile = np.zeros((Np, 2))

    for j in range(Np):
        if i+j >= len(tValues):
            prediction_road_profile[j] = np.array([0, 0])
        else:
            prediction_road_profile[j] = np.array([road_profile_derivative_front[i+j], road_profile_derivative_rear[i+j]])

    # for j in range(Np):
    #     index = i + j * int(dt_prediction / dt)
    #     if index >= len(tValues):
    #         prediction_road_profile[j] = np.array([0, 0])
    #     else:
    #         prediction_road_profile[j] = np.array([road_profile_derivative_front[index], road_profile_derivative_rear[index]])

    # solve for the control input
    u = quarter_car(par, Np, dt_prediction, state, prediction_road_profile[:, 0], prediction_road_profile[:, 1])
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
    change_passive = dt* derivative_passive
    state_passive = state_passive + np.array([[change_passive[0, 0]], [change_passive[2, 0]], [0], [0], [change_passive[1, 0]], [change_passive[3, 0]], [0], [0]])

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

plt.figure(figsize=(15, 15))
plt.subplot(7, 1, 1)
plt.plot(tValues, state_history[:, 0], label='Front suspension deflection')
plt.plot(tValues, passive_state[:, 0], label='Front suspension deflection passive')
# plt.plot(tValues, state_history[:, 2], label='Rear suspension deflection')
plt.plot(tValues, state_history[:, 4], label='Front tire deflection')
plt.plot(tValues, passive_state[:, 4], label='Front tire deflection passive')
plt.axhline(0, linestyle='--')
# plt.plot(tValues, state_history[:, 6], label='Rear tire deflection')
plt.legend()

plt.subplot(7, 1, 2)
plt.plot(tValues, acceleration_history[:, 0], label='Body acceleration')
plt.plot(tValues, passive_acceleration[:, 0], label='Body acceleration passive')
# plt.plot(tValues, acceleration_history[:, 1], label='Pitch acceleration')
plt.axhline(0, linestyle='--')
plt.legend()

plt.subplot(7, 1, 3)
plt.plot(tValues, state_history[:, 1], label='Front suspension deflection speed')
plt.plot(tValues, passive_state[:, 1], label='Front suspension deflection speed passive')
# plt.plot(tValues, state_history[:, 3], label='Rear suspension deflection speed')
plt.plot(tValues, state_history[:, 5], label='Front tire deflection speed')
plt.plot(tValues, passive_state[:, 5], label='Front tire deflection speed passive')
# plt.plot(tValues, state_history[:, 7], label='Rear tire deflection speed')
plt.axhline(0, linestyle='--')
plt.legend()

plt.subplot(7, 1, 4)
plt.plot(tValues, u_history[:, 0], label='Input front')
plt.axhline(0, linestyle='--')

plt.subplot(7, 1, 5)
plt.plot(tValues, damping_force_history, label='Damping force')
plt.plot(tValues, passive_damping_force, label='Damping force passive')
plt.axhline(0, linestyle='--')

plt.subplot(7, 1, 6)
plt.plot(tValues, z_values, label='zs - zu vel')
plt.plot(tValues, passive_z_values, label='zs - zu vel passive')
plt.axhline(0, linestyle='--')

plt.subplot(7, 1, 7)
plt.plot(tValues, road_profile_front[0:-1], label='Road profile')

plt.figure(figsize=(15, 15))
plt.scatter(z_values, damping_force_history, label='Damping force')
plt.scatter(passive_z_values, passive_damping_force, label='Damping force passive')


# plt.plot(tValues, acceleration_history[:, 0], label='Body acceleration')
# plt.plot(tValues, acceleration_history[:, 1], label='Pitch acceleration')
# plt.plot(tValues, state_history[:, 1], label='Front suspension deflection speed')
# plt.plot(tValues, state_history[:, 2], label='Rear suspension deflection')
# plt.plot(tValues, state_history[:, 4], label='Front tire deflection')
# plt.plot(tValues, state_history[:, 6], label='Rear tire deflection')
# plt.plot(tValues, road_profile_front[0:-1], label='Road profile')
plt.legend()
# plt.xlim([3, 4])
plt.show()
