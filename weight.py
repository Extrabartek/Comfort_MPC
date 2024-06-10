import numpy as np
import matplotlib.pyplot as plt
from control.matlab import bode, tf

# Define the frequency points
n_points = int(10e5)
f = np.linspace(0.01, 1000, n_points)
w = f * 2 * np.pi

# Define the transfer functions using tf
Wv = tf([87.72, 1138, 11336, 5452, 5509], [1, 92.6854, 2549.83, 25969, 81057, 79783])
Wh = tf([12.66, 163.7, 60.04, 12.79], [1, 23.77, 236.1, 692.8, 983.4])
Wm = tf([0.1457, 0.2331, 13.75, 1.705, 0.3596], [1, 7.757, 19.06, 28.37, 18.52, 7.230])

# Calculate magnitude and phase
magWeightVertical, phaseVertical, omega = bode(Wv, w, plot=False)
magWeightHorizontal, phaseHorizontal, omega = bode(Wh, w, plot=False)
magWeightMotion, phaseMotion, omega = bode(Wm, w, plot=False)

# Print sizes and values
print(f"size of omega: {omega.size}")
print(f"val of magWeightVertical: {magWeightVertical}")
print(f"size of magWeightHorizontal: {magWeightHorizontal.size}")
print(f"size of magWeightMotion: {magWeightMotion.size}")

# Plot the results
plt.figure()
plt.semilogx(omega / (2 * np.pi), 20 * np.log10(magWeightVertical), label='Vertical')
plt.semilogx(omega / (2 * np.pi), 20 * np.log10(magWeightHorizontal), label='Horizontal')
plt.semilogx(omega / (2 * np.pi), 20 * np.log10(magWeightMotion), label='Motion')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.legend()
plt.grid(which='both', linestyle='-', color='grey', linewidth=0.5)
plt.show()


# Low order weight filters
# Wv_low1 = (50 * s + 500) / (s**2 + 50 * s + 1200)
Wv_low1 = tf([50, 500], [1, 50, 1200])
# Wv_low2 = (86.51 * s + 546.1) / (s**2 + 82.17 * s + 1892)
Wv_low2 = tf([86.51, 546.1], [1, 82.17, 1892])
# Wh_low = (13.55 * s) / (s**2 + 12.90 * s + 47.16)
Wh_low = tf([13.55, 0], [1, 12.90, 47.16])
# Wm_low = (0.8892 * s) / (s**2 + 0.8263 * s + 1.163)
Wm_low = tf([0.8892, 0], [1, 0.8263, 1.163])

# Calculate magnitude and phase
magWeightVerticalLow1, phaseWeightVerticalLow1, omega = bode(Wv_low1, w, plot=False)
magWeightVerticalLow2, phaseWeightVerticalLow2, omega = bode(Wv_low2, w, plot=False)
magWeightHorizontalLow, phaseWeightHorizontalLow, omega = bode(Wh_low, w, plot=False)
magWeightMotionLow, phaseWeightMotionLow, omega = bode(Wm_low, w, plot=False)

# plot the results
plt.figure()
plt.semilogx(omega / (2 * np.pi), 20 * np.log10(magWeightVerticalLow1), label='Vertical Low 1')
plt.semilogx(omega / (2 * np.pi), 20 * np.log10(magWeightVerticalLow2), label='Vertical Low 2')
plt.semilogx(omega / (2 * np.pi), 20 * np.log10(magWeightHorizontalLow), label='Horizontal Low')
plt.semilogx(omega / (2 * np.pi), 20 * np.log10(magWeightMotionLow), label='Motion Low')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.legend()
plt.grid(which='both', linestyle='-', color='grey', linewidth=0.5)
plt.show()