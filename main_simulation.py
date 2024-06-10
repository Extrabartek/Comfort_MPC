import numpy as np
import matplotlib.pyplot as plt

from roadsurface import isolatedBump, isolatedTable, isoRoad
from state_space_half_car import half_car_state_space, Parameters

par = Parameters(960, 1222, 40, 45, 200000,
                 200000, 18000, 22000, 1000,
                 1000, 1.3, 1.5)

# Define the state-space matrices
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
f = 10000  # Hz
endTime = 10  # s
tValues = np.arange(0, endTime, 1 / f)  # the time array [s]

# Tunable parameters (dependent on bump profile)
A = 1  # amplitude of the bump [m]
V = 75 / 3.6  # velocity of the car [m/s]
l = 5  # length of the bump [m]
L = 60  # length until the bump [m]

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
n = len(tValues)  # number of samples
state_history = np.zeros((n, 8))  # state history
derivative_history = np.zeros((n, 8))  # derivative history
acceleration_history = np.zeros((n, 2))  # acceleration history

# get the state-spaces matrices
A, B, F, C, E = half_car_state_space(par, 1, 1)

for i in range(n):
    # get the current road profile
    road_profile = np.array([road_profile_front[i], road_profile_rear[i]])

    # calculate the acceleration
    derivative = np.dot(A, state.T) + np.dot(B, road_profile.T) + np.dot(F, np.array([0, 0]).T)

    # calculate the derivative
    acceleration = np.dot(C, state.T) + np.dot(E, road_profile.T)

    # update the state
    state = state + dt * derivative

    # save the state
    state_history[i] = state
    derivative_history[i] = derivative
    acceleration_history[i] = acceleration


plt.plot(tValues, acceleration_history[:, 0], label='Body acceleration')

plt.show()