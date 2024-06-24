import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import freqresp
from scipy.signal import TransferFunction as tf
import matplotlib.pyplot as plt
from metrics import get_a_w, rms
import pickle as pkl
from metrics import wrms, wrmq


def plot_spectrum_comparison(a_z):
    a_w, w = get_a_w(a_z)
    A_z = fft(a_z)
    A_w = fft(a_w)

    plt.figure()
    plt.plot(w, np.abs(A_z), label='Input')
    plt.plot(w, np.abs(A_w), label='Weighted')
    plt.xlabel('Frequency [rad/s]')
    plt.ylabel('Magnitude')
    plt.title('Input and Weighted Acceleration in Frequency Domain')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_acceleration_comparison(ts, a_z):
    '''Plot the input and weighted acceleration in the time domain'''
    a_w, w = get_a_w(a_z)

    plt.figure()
    plt.plot(ts, a_z, label='Input')
    plt.plot(ts, a_w, label='Weighted')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.title('Input and Weighted Acceleration in Time Domain')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_quarter(name: str):
    """ Plot the results of the simulation

    Parameters:
        name (str): The name of the file containing the results
    """
    try:
        # Load the results
        with open("results/" + name, 'rb') as file:
            state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax = pkl.load(
                file)
            print("Results loaded successfully")
    except Exception as e:
        print(f"Error: {e}")

    print(tValues.size, state_history[:, 0].size)

    plt.figure(figsize=(4, 3))
    sus_deflection = np.fft.fft(state_history[:, 0])
    sus_deflection_passive = np.fft.fft(state_pass_history[:, 0])
    freq = np.fft.fftfreq(state_history[:, 0].size, d=tValues[1] - tValues[0])
    sus_deflection = np.abs(sus_deflection)

    print(freq)
    print(sus_deflection)
    print(len(freq), len(sus_deflection))
    plt.plot(freq, np.abs(sus_deflection), label='MPC')
    plt.plot(freq, np.abs(sus_deflection_passive), label='Passive')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')


    plt.figure(figsize=(12, 12))
    # plt.plot(tValues, state_history[:, 0], label='Front suspension deflection')
    # plt.plot(tValues, passive_state[:, 0], label='Front suspension deflection passive')
    # # plt.plot(tValues, state_history[:, 2], label='Rear suspension deflection')
    # plt.plot(tValues, state_history[:, 4], label='Front tire deflection')
    # plt.plot(tValues, passive_state[:, 4], label='Front tire deflection passive')
    # plt.axhline(0, linestyle='--')
    # # plt.plot(tValues, state_history[:, 6], label='Rear tire deflection')
    # plt.legend()

    plt.subplot(4, 1, 1)
    plt.plot(tValues, output_history[:, 0], label='Body acceleration')
    plt.plot(tValues, output_pass_history[:, 0], label='Body acceleration passive')
    # # plt.plot(tValues, acceleration_history[:, 1], label='Pitch acceleration')
    # plt.axhline(0, linestyle='--')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Acceleration [m/s^2]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    #
    # plt.subplot(7, 1, 3)
    # plt.plot(tValues, state_history[:, 1], label='Front suspension speed')
    # plt.plot(tValues, passive_state[:, 1], label='Front suspension speed passive')
    # # plt.plot(tValues, state_history[:, 3], label='Rear suspension deflection speed')
    # plt.plot(tValues, state_history[:, 5], label='Front tire speed')
    # plt.plot(tValues, passive_state[:, 5], label='Front tire speed passive')
    # # plt.plot(tValues, state_history[:, 7], label='Rear tire deflection speed')
    # plt.axhline(0, linestyle='--')
    # plt.legend()
    #
    plt.subplot(4, 1, 2)
    plt.plot(tValues, u_history[:, 0], label='Control Input')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Force [N]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    #
    plt.subplot(4, 1, 3)
    plt.plot(tValues, damping_force_history, label='Total damping force - Active Damper')
    # plt.plot(tValues, damping_force_history - u_history[:, 0], label='Damping force - input front')
    plt.plot(tValues, damping_force_passive, label='Total damping force - Passive Damper')
    # plt.axhline(0, linestyle='--')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Force [N]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    #
    # plt.subplot(7, 1, 6)
    # plt.plot(tValues, z_values, label='zs - zu vel')
    # plt.plot(tValues, passive_z_values, label='zs - zu vel passive')
    # plt.axhline(0, linestyle='--')
    # plt.legend()
    #
    plt.subplot(4, 1, 4)
    plt.plot(tValues, road_profile_front[0:-1], label='Road profile')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Displacement [m]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)

    plt.figure(figsize=(10, 10))
    plt.scatter(deflection_velocity_passive, damping_force_passive, label='Passive Damper', color='red')
    plt.scatter(deflection_velocity, damping_force_history, label='Active Damper')
    # need to add parameters as a saved value
    z_values_range = np.linspace(np.min(deflection_velocity), np.max(deflection_velocity), num=len(deflection_velocity))
    plt.plot(z_values_range, z_values_range*csf, linestyle='--', color='black', label='Passive Damper')
    plt.plot(z_values_range, z_values_range*csmax, linestyle='--', color='red', label='Max Active Damper')
    plt.plot(z_values_range, z_values_range*csmin, linestyle='--', color='blue', label='Min Active Damper')
    plt.xlabel('Suspension Velocity [m/s]', fontsize=20)
    plt.ylabel('Damping Force [N]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.legend(fontsize=20)
    plt.tight_layout()
    # plt.show()
    ##################################################
    # plt.figure()
    # plt.plot(tValues, z_values, label='zs - zu vel')
    # plt.plot(tValues, passive_z_values, label='zs - zu vel passive')
    # plt.plot(tValues, delta_front, label="delta front", marker=".")
    # plt.plot(tValues, delta_rear, label="delta rear", marker=".")
    # plt.axhline(0, linestyle='--')
    # plt.legend()
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(tValues, damping_force_history, label='Damping force')
    # plt.plot(tValues, zlistf, label='z')
    # plt.plot(tValues, np.array(delta_front) * 4000, label="delta front", marker=".")
    # plt.axhline(0, linestyle='--')
    # plt.legend()
    # plt.grid()
    # plt.show()
    plt.tight_layout()
    plt.show()

    print(wrms([], output_history[:, 0]))
    print(wrms([], output_pass_history[:, 0]))

    print(wrmq(output_history[:, 0], []))
    print(wrmq(output_pass_history[:, 0], []))


if __name__ == "__main__":
    plot_quarter("results_type_bump_endT_0.2_f_1000_tl_0.02_Np_10_quarter.pkl")

