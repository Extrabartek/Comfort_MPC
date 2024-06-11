import numpy as np
import matplotlib.pyplot as plt

from roadsurface import isolatedBump, isolatedTable, isoRoad
from state_space_half_car import half_car_state_space, Parameters
from quarter import quarter_car

par = Parameters(960, 1222, 40, 45, 200000,
                 200000, 18000, 22000, 1000,
                 1000, 1.3, 1.5)

# Define the state-space matrices
state = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

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
tl = 0.01  # time of the bump [s]
l = tl * V  # position of the bump [m]
L = 0.5  # length of the bump [m]

# run the script to create the road profile
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
derivative_history = np.zeros((n, 8, 1))  # derivative history
acceleration_history = np.zeros((n, 2, 1))  # acceleration history

# generate road profile derivatives
road_profile_derivative_front = np.gradient(road_profile_front, dt)
road_profile_derivative_rear = np.gradient(road_profile_rear, dt)

# get the state-spaces matrices
A, B, F, C, E = half_car_state_space(par)

for i in range(n):
    # create the road profile based on the derivative
    road_profile = np.array([[road_profile_derivative_front[i]], [road_profile_derivative_rear[i]]])

    # create the road profile for the prediction horizon
    prediction_road_profile = np.zeros((Np, 2))
    for j in range(Np):
        index = i + j * int(dt_prediction / dt)
        if index >= len(tValues):
            prediction_road_profile[j] = np.array([0, 0])
        else:
            prediction_road_profile[j] = np.array([road_profile_front[index], road_profile_rear[index]])

    # solve for the control input
    u = quarter_car(par, Np, dt_prediction, state, prediction_road_profile[:, 0], prediction_road_profile[:, 1])
    u = np.array([[u[0]], [u[1]]])
    # u = np.array([[0], [0]])
    # calculate the derivative
    derivative = A @ state + B @ road_profile + F @ u

    # calculate the acceleration
    acceleration = C @ state + E @ road_profile

    # update the state
    state = state + dt * derivative

    # save the state
    state_history[i] = state
    derivative_history[i] = derivative
    acceleration_history[i] = acceleration
    print(f"Step {i} of {n}")

# create seperated sub-figures for the acceleration, the state and the road profile

plt.figure(figsize=(15, 15))
plt.subplot(3, 1, 1)
plt.plot(tValues, state_history[:, 0], label='Front suspension deflection')
plt.plot(tValues, state_history[:, 2], label='Rear suspension deflection')
plt.plot(tValues, state_history[:, 4], label='Front tire deflection')
plt.plot(tValues, state_history[:, 6], label='Rear tire deflection')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(tValues, acceleration_history[:, 0], label='Body acceleration')
plt.plot(tValues, acceleration_history[:, 1], label='Pitch acceleration')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(tValues, state_history[:, 1], label='Front suspension deflection speed')
plt.plot(tValues, state_history[:, 3], label='Rear suspension deflection speed')
plt.plot(tValues, state_history[:, 5], label='Front tire deflection speed')
plt.plot(tValues, state_history[:, 7], label='Rear tire deflection speed')
plt.legend()

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
