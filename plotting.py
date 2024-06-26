import control
import numpy as np
import scipy.fft as ft
from scipy.fft import fft, ifft
from scipy.signal import freqresp
from scipy.signal import TransferFunction as tf
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
from metrics import get_a_w, rms, wrms
import pickle as pkl
from metrics import wrms, wrmq

# Other styles
# plt.style.use('ggplot')
# plt.style.use('tableau-colorblind10')
plt.style.use('seaborn-v0_8-paper')
# print(plt.style.available)

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
    a_w, w = get_a_w(a_z, 1/(ts[1]-ts[0]))

    plt.figure()
    plt.plot(ts, a_z, label='Input')
    plt.plot(ts, a_w, label='Weighted')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.title('Input and Weighted Acceleration in Time Domain')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_half(name:str):
    """ Plot the results of the simulation

    Parameters:
        name (str): The name of the file containing the results
    """
    try:
        # Load the results
        with open("results/" + name, 'rb') as file:
            state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax, par, state_quarter_history, output_quarter_history, u_quarter_history, damping_force_quarter, deflection_velocity_quarter = pkl.load(
                file)
            print("Results loaded successfully")
    except Exception as e:
        print(f"Error: {e}")

    active_wrms = wrms(output_history[:, 0], 1/(tValues[1]-tValues[0]))
    passive_wrms = wrms(output_pass_history[:, 0], 1/(tValues[1]-tValues[0]))
    quarter_wrms = wrms(output_quarter_history[:, 0], 1/(tValues[1]-tValues[0]))

    print(f"Half wrms: {active_wrms}")
    print(f"Quarter wrms: {quarter_wrms}")
    print(f"Passive wrms: {passive_wrms}")

    print(f"The percentage improvement in WRMS is: {100 * (passive_wrms - active_wrms) / passive_wrms} %")

    active_wrms = wrmq(output_history[:, 0], 1/(tValues[1]-tValues[0]))
    passive_wrms = wrmq(output_pass_history[:, 0], 1/(tValues[1]-tValues[0]))
    quarter_wrms = wrmq(output_quarter_history[:, 0], 1/(tValues[1]-tValues[0]))

    print(f"The percentage improvement in WRMQ is {100 * (passive_wrms - active_wrms) / passive_wrms} %")

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD: Body acceleration [(m/s^2)^2/Hz] ')
    freq_psd, result_psd = signal.periodogram(output_pass_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_active = signal.periodogram(output_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_quarter = signal.periodogram(output_quarter_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    plt.loglog(freq_psd, result_psd, label='Passive PSD')
    plt.loglog(freq_psd, result_psd_active, label='Active PSD')
    plt.loglog(freq_psd, result_psd_quarter, label='Quarter PSD')
    plt.legend()
    plt.xlim([1e-2, 25])
    plt.hlines(1, 0, 1000, linestyle='--')
    plt.ylim([1e-9, 1e1])
    plt.grid()
    plt.title('This should work')


    plt.figure(figsize=(12, 12))
    # plt.plot(tValues, state_history[:, 0], label='Front suspension deflection')
    # plt.plot(tValues, passive_state[:, 0], label='Front suspension deflection passive')
    # # plt.plot(tValues, state_history[:, 2], label='Rear suspension deflection')
    # plt.plot(tValues, state_history[:, 4], label='Front tire deflection')
    # plt.plot(tValues, passive_state[:, 4], label='Front tire deflection passive')
    # plt.axhline(0, linestyle='--')
    # # plt.plot(tValues, state_history[:, 6], label='Rear tire deflection')
    # plt.legend()

    plt.subplot(6, 1, 1)
    plt.plot(tValues, output_history[:, 0], label='Body acceleration')
    plt.plot(tValues, output_pass_history[:, 0], label='Body acceleration passive')
    plt.plot(tValues, output_quarter_history[:, 0], label='Body acceleration quarter')
    # # plt.plot(tValues, acceleration_history[:, 1], label='Pitch acceleration')
    # plt.axhline(0, linestyle='--')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Acceleration [m/s^2]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    #

    plt.subplot(6, 1, 2)
    plt.plot(tValues, output_history[:, 1], label='Pitch acceleration')
    plt.plot(tValues, output_pass_history[:, 1], label='Pitch acceleration passive')
    plt.plot(tValues, output_quarter_history[:, 1], label='Pitch acceleration quarter')
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(tValues, state_history[:, 1] - state_history[:, 5], label='Front suspension speed')
    plt.plot(tValues, state_pass_history[:, 1] - state_pass_history[:, 5], label='Front suspension speed passive')
    plt.plot(tValues, state_quarter_history[:, 1] - state_quarter_history[:, 5], label='Front suspension speed quarter')
    plt.axhline(0, linestyle='--')
    plt.legend()
    #
    plt.subplot(6, 1, 4)
    plt.plot(tValues, u_history[:, 0], label='Control Input Front')
    plt.plot(tValues, u_history[:, 1], label='Control Input Rear')
    plt.plot(tValues, u_quarter_history[:, 0], label='Control Quarter Front')
    plt.plot(tValues, u_quarter_history[:, 1], label='Control Quarter Rear')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Force [N]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    #
    plt.subplot(6, 1, 5)
    plt.plot(tValues, damping_force_history, label='Total damping force - Active Damper')
    # plt.plot(tValues, damping_force_history - u_history[:, 0], label='Damping force - input front')
    plt.plot(tValues, damping_force_passive, label='Total damping force - Passive Damper')
    plt.plot(tValues, damping_force_quarter, label='Total damping force - Quarter Damper')
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
    plt.subplot(6, 1, 6)
    try:
        plt.plot(tValues, road_profile_front[0:-1], label='Road profile')
    except:
        plt.plot(tValues, road_profile_front[0:-2], label='Road profile')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Displacement [m]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)

    plt.figure(figsize=(10, 10))
    plt.scatter(deflection_velocity_passive, damping_force_passive, label='Passive Damper', color='red')
    plt.scatter(deflection_velocity, damping_force_history, label='Active Damper')
    plt.scatter(deflection_velocity_quarter, damping_force_quarter, label='Quarter Damper')
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

def plot_quarter(name: str):
    """ Plot the results of the simulation

    Parameters:
        name (str): The name of the file containing the results
    """
    try:
        # Load the results
        with open("results/" + name, 'rb') as file:
            state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax, par = pkl.load(
                file)
            print("Results loaded successfully")
    except Exception as e:
        print(f"Error: {e}")

    active_wrms = wrms(output_history[:, 0], 1/(tValues[1]-tValues[0]))
    passive_wrms = wrms(output_pass_history[:, 0], 1/(tValues[1]-tValues[0]))

    print(f"WRMS: {active_wrms}, Passive: {passive_wrms}, diff: {100 * (passive_wrms - active_wrms) / passive_wrms} %")

    active_wrmq = wrmq(output_history[:, 0], 1/(tValues[1]-tValues[0]))
    passive_wrmq = wrmq(output_pass_history[:, 0], 1/(tValues[1]-tValues[0]))

    print(f"WRMQ: {active_wrmq}, Passive: {passive_wrmq}, diff: {100 * (passive_wrmq - active_wrmq) / passive_wrmq} %")

    road_rms = rms(output_history[:, 1])
    road_passive_rms = rms(output_pass_history[:, 1])

    print(f"RMS: {road_rms}, Passive: {road_passive_rms}, diff: {100 * (road_passive_rms - road_rms) / road_passive_rms} %")

    ##################################################
    # PSD BODY ACCELERATION
    plt.figure()
    plt.xlabel(r'Frequency [$Hz$]', fontsize=11)
    plt.ylabel(r"PSD: Body acceleration [$(m/s^2)^2Hz$]", fontsize=11)
    freq_psd, result_psd = signal.periodogram(output_pass_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_active = signal.periodogram(output_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    plt.loglog(freq_psd, result_psd, label='Passive PSD')
    plt.loglog(freq_psd, result_psd_active, label='Active PSD')
    # freq_psd, result_psd = signal.welch(output_pass_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    # freq_psd, result_psd_active = signal.welch(output_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    # plt.loglog(freq_psd, result_psd, label='Passive')
    # plt.loglog(freq_psd, result_psd_active, label='Active')
    freq_psd, result_psd = signal.periodogram(get_a_w(output_pass_history[:, 0].ravel(), 1/(tValues[1]-tValues[0]))[0],
                                              fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_active = signal.periodogram(get_a_w(output_history[:, 0].ravel(), 1/(tValues[1]-tValues[0]))[0],
                                                     fs=1 / (tValues[1] - tValues[0]))
    plt.loglog(freq_psd, result_psd, label='Weighted Passive PSD')
    plt.loglog(freq_psd, result_psd_active, label='Weighted Active PSD')
    # freq_psd, result_psd = signal.welch(get_a_w(output_pass_history[:, 0].ravel(), 1/(tValues[1]-tValues[0]))[0],
    #                                     fs=1 / (tValues[1] - tValues[0]))
    # freq_psd, result_psd_active = signal.welch(get_a_w(output_history[:, 0].ravel(), 1/(tValues[1]-tValues[0]))[0],
    #                                               fs=1 / (tValues[1] - tValues[0]))
    # plt.loglog(freq_psd, result_psd, label='Weighted Passive')
    # plt.loglog(freq_psd, result_psd_active, label='Weighted Active')
    plt.legend()
    plt.xlim([0.4, 16])
    plt.hlines(1, 0, 1000, linestyle='--', colors='black')
    plt.ylim([1e-4, 5e0])
    plt.grid()

    ##################################################
    # PSD BODY ACCELERATION WELCH
    plt.figure()
    plt.xlabel(r'Frequency [$Hz$]', fontsize=11)
    plt.ylabel(r"PSD: Body acceleration [$(m/s^2)^2Hz$]", fontsize=11)
    # freq_psd, result_psd = signal.periodogram(output_pass_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    # freq_psd, result_psd_active = signal.periodogram(output_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    # plt.loglog(freq_psd, result_psd, label='Passive PSD')
    # plt.loglog(freq_psd, result_psd_active, label='Active PSD')
    freq_psd, result_psd = signal.welch(output_pass_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_active = signal.welch(output_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    plt.loglog(freq_psd, result_psd, label='Passive')
    plt.loglog(freq_psd, result_psd_active, label='Active')
    # freq_psd, result_psd = signal.periodogram(get_a_w(output_pass_history[:, 0].ravel(), 1/(tValues[1]-tValues[0]))[0],
    #                                           fs=1 / (tValues[1] - tValues[0]))
    # freq_psd, result_psd_active = signal.periodogram(get_a_w(output_history[:, 0].ravel(), 1/(tValues[1]-tValues[0]))[0],
    #                                                  fs=1 / (tValues[1] - tValues[0]))
    # plt.loglog(freq_psd, result_psd, label='Weighted Passive PSD')
    # plt.loglog(freq_psd, result_psd_active, label='Weighted Active PSD')
    freq_psd, result_psd = signal.welch(get_a_w(output_pass_history[:, 0].ravel(), 1/(tValues[1]-tValues[0]))[0],
                                        fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_active = signal.welch(get_a_w(output_history[:, 0].ravel(), 1/(tValues[1]-tValues[0]))[0],
                                                  fs=1 / (tValues[1] - tValues[0]))
    plt.loglog(freq_psd, result_psd, label='Weighted Passive')
    plt.loglog(freq_psd, result_psd_active, label='Weighted Active')
    plt.legend()
    plt.xlim([0.4, 16])
    plt.hlines(1, 0, 1000, linestyle='--', colors='black')
    plt.ylim([1e-4, 5e0])
    plt.grid()

    ##################################################
    # PSD ROAD HOLDING
    plt.figure()
    plt.xlabel(r'Frequency [$Hz$]', fontsize=11)
    plt.ylabel(r'PSD: Road holding [$m^2/Hz$]', fontsize=11)
    # freq_psd, result_psd = signal.periodogram(output_pass_history[:, 1].ravel(), fs=1 / (tValues[1] - tValues[0]))
    # freq_psd, result_psd_active = signal.periodogram(output_history[:, 1].ravel(), fs=1 / (tValues[1] - tValues[0]))
    # plt.loglog(freq_psd, result_psd, label='Passive PSD')
    # plt.loglog(freq_psd, result_psd_active, label='Active PSD')
    freq_psd, result_psd = signal.welch(output_pass_history[:, 1].ravel(), fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_active = signal.welch(output_history[:, 1].ravel(), fs=1 / (tValues[1] - tValues[0]))
    plt.loglog(freq_psd, result_psd, label='Passive')
    plt.loglog(freq_psd, result_psd_active, label='Active')
    plt.legend()
    plt.xlim([0.4, 16])
    plt.ylim([1e-9, 5e-6])
    plt.grid()

    ##################################################
    # TIME
    plt.figure(figsize=(12, 12))
    # plt.plot(tValues, state_history[:, 0], label='Front suspension deflection')
    # plt.plot(tValues, passive_state[:, 0], label='Front suspension deflection passive')
    # # plt.plot(tValues, state_history[:, 2], label='Rear suspension deflection')
    # plt.plot(tValues, state_history[:, 4], label='Front tire deflection')
    # plt.plot(tValues, passive_state[:, 4], label='Front tire deflection passive')
    # plt.axhline(0, linestyle='--')
    # # plt.plot(tValues, state_history[:, 6], label='Rear tire deflection')
    # plt.legend()

    plt.subplot(6, 1, 1)
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

    plt.subplot(6, 1, 2)
    plt.plot(tValues, output_history[:, 1], label='Pitch acceleration')
    plt.plot(tValues, output_pass_history[:, 1], label='Pitch acceleration passive')
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(tValues, state_history[:, 1] - state_history[:, 5], label='Front suspension speed')
    plt.plot(tValues, state_pass_history[:, 1] - state_pass_history[:, 5], label='Front suspension speed passive')
    plt.axhline(0, linestyle='--')
    plt.legend()
    #
    plt.subplot(6, 1, 4)
    plt.plot(tValues, u_history[:, 0], label='Control Input')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Force [N]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    #
    plt.subplot(6, 1, 5)
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
    plt.subplot(6, 1, 6)
    try:
        plt.plot(tValues, road_profile_front[0:-1], label='Road profile')
    except:
        plt.plot(tValues, road_profile_front[0:-2], label='Road profile')
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Displacement [m]', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)

    # time trace simplified
    time_trace_length = 150
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(tValues[:time_trace_length], output_history[:, 0][:time_trace_length], label='Body acceleration')
    plt.plot(tValues[:time_trace_length], output_pass_history[:, 0][:time_trace_length], label='Body acceleration passive')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    try:
        plt.plot(tValues[:time_trace_length], road_profile_front[0:-1][:time_trace_length], label='Road profile')
    except:
        plt.plot(tValues[:time_trace_length], road_profile_front[0:-2][:time_trace_length], label='Road profile')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.grid()
    plt.legend()

    # time trace apendix
    time_trace_length = 150
    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1)
    plt.plot(tValues[:time_trace_length], output_history[:, 0][:time_trace_length], label='Body acceleration')
    plt.plot(tValues[:time_trace_length], output_pass_history[:, 0][:time_trace_length], label='Body acceleration passive')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(tValues[:time_trace_length], u_history[:, 0][:time_trace_length], label='Control Input')
    plt.plot(tValues[:time_trace_length], damping_force_history[:time_trace_length], label='Total damping force - Active Damper')
    # plt.plot(tValues[:time_trace_length], damping_force_passive[:time_trace_length], label='Total damping force - Passive Damper')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    try:
        plt.plot(tValues[:time_trace_length], road_profile_front[0:-1][:time_trace_length], label='Road profile')
    except:
        plt.plot(tValues[:time_trace_length], road_profile_front[0:-2][:time_trace_length], label='Road profile')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.grid()
    plt.legend()




    #########################################
    # Actuator scatter plot
    plt.figure()
    plt.scatter(deflection_velocity, damping_force_history, label='Active', color="#1192e8", alpha=0.75, s=15)
    plt.scatter(deflection_velocity_passive, damping_force_passive, label='Passive', color="#da1e28", marker="D", alpha=0.95, s=15)
    z_values_range = np.linspace(np.min(deflection_velocity)*10, np.max(deflection_velocity)*10, num=len(deflection_velocity))
    plt.plot(z_values_range, z_values_range*csf, linestyle='-', color='#1c0f30', label='Nominal Damper', linewidth=1.5)
    plt.plot(z_values_range, z_values_range*csmax, linestyle='-.', color='#1c0f30', label='Active Damper Envelope', linewidth=1.5)
    plt.plot(z_values_range, z_values_range*csmin, linestyle='-.', color='#1c0f30')
    plt.xlabel(r'Suspension Velocity [$m/s$]', fontsize=11)
    plt.ylabel(r'Damping Force [$N$]', fontsize=11)
    plt.ylim([damping_force_history.min()*1.1, damping_force_history.max()*1.1])
    plt.xlim([deflection_velocity.min()*1.1, deflection_velocity.max()*1.1])
    plt.grid()
    plt.legend()
    

    plt.show()

def plot_different_np(files: list[str]):
    results = []
    for file_name in files:
        try:
            # Load the results
            with open("results/" + file_name, 'rb') as file:
                state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax, par = pkl.load(
                    file)
        except Exception as e:
            print(f"Error: {e}")

        result = {"state_history": state_history, "output_history": output_history, "u_history": u_history,
                  "road_profile_front": road_profile_front, "road_profile_rear": road_profile_rear,
                  "damping_force_history": damping_force_history, "deflection_velocity": deflection_velocity,
                  "damping_force_passive": damping_force_passive,
                  "deflection_velocity_passive": deflection_velocity_passive, "tValues": tValues,
                  "state_pass_history": state_pass_history, "output_pass_history": output_pass_history, "csf": csf,
                  "csr": csr, "csmin": csmin, "csmax": csmax, "par": par}
        results.append(result)
        print("Results loaded successfully")

    plt.figure()
    plt.xlabel(r'Frequency [$Hz$]', fontsize=11)
    plt.ylabel(r"PSD: Body acceleration [$(m/s^2)^2Hz$]", fontsize=11)
    fs = 1 / (results[0]['tValues'][1] - results[0]['tValues'][0])
    freq_psd, psd_pass = signal.welch(get_a_w(results[0]['output_pass_history'][:, 0].ravel(), fs=fs)[0], fs=fs)
    plt.loglog(freq_psd, psd_pass, label='Passive')
    freq_psd, psd_2 = signal.welch(get_a_w(results[0]['output_history'][:, 0].ravel(), fs=fs)[0], fs=fs)
    plt.loglog(freq_psd, psd_2, label='Np = 2')
    freq_psd, psd_10 = signal.welch(get_a_w(results[1]['output_history'][:, 0].ravel(), fs=fs)[0], fs=fs)
    plt.loglog(freq_psd, psd_10, label='Np = 10')
    freq_psd, psd_20 = signal.welch(get_a_w(results[2]['output_history'][:, 0].ravel(), fs=fs)[0], fs=fs)
    plt.loglog(freq_psd, psd_20, label='Np = 20')
    plt.xlim([0.4, 16])
    plt.hlines(1, 0, 1000, linestyle='--', colors='black')
    plt.legend()
    plt.ylim([1e-4, 5e0])
    plt.grid()

def regenerate_D_results():
    paraWeight = []
    paraComfort = []
    paraComfortWeighted = []
    paraHolding = []

    files = ['0.01', 
             '500',
             '3000',
             '8771.939649122807',
             '17543.869298245612',
             '26315.798947368417', 
             '39473.69342105263',
             '52631.58789473684', 
             '78947.37684210525', 
             '105263.16578947367', 
             '131578.9547368421', 
             '157894.74368421052', 
             '184210.53263157894', 
             '210526.32157894736', 
             '236842.11052631578', 
             '263157.8994736842', 
             '289473.68842105265', 
             '315789.47736842104', 
             '342105.2663157894', 
             '368421.0552631579', 
             '394736.8442105263', 
             '421052.6331578947', 
             '447368.4221052631', 
             '473684.21105263155', 
             '500000.0']
    for f in files:
        with open(f"results/road_D_25kph_30sec_30Hz/time_traces/results_w1_1_w2_{f}.pkl", 'rb') as file:
            state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax, par = pkl.load(
                file)

        paraWeight.append(float(f))
        paraComfort.append(rms(output_history[:, 0]))
        paraHolding.append(rms(output_history[:, 1]))
        paraComfortWeighted.append(wrms(output_history[:, 0], 1/(tValues[1]-tValues[0])))

    results = [paraWeight, paraComfort, paraHolding, paraComfortWeighted]

    with open('results/road_D_25kph_30sec_30Hz/results_weightSens.pkl', 'wb') as f:
        pkl.dump(results, f)

def regenerate_A_results():
    paraWeight = []
    paraComfort = []
    paraComfortWeighted = []
    paraHolding = []

    files = ['0.01', 
             '8771.939649122807',
             '17543.869298245612',
             '26315.798947368417',
             '39473.69342105263', 
             '52631.58789473684', 
             '78947.37684210525', 
             '105263.16578947367', 
             '131578.9547368421', 
             '157894.74368421052', 
             '184210.53263157894', 
             '210526.32157894736', 
             '236842.11052631578', 
             '263157.8994736842', 
             '289473.68842105265', 
             '315789.47736842104', 
             '342105.2663157894', 
             '368421.0552631579', 
             '394736.8442105263', 
             '421052.6331578947', 
             '447368.4221052631', 
             '473684.21105263155', 
             '500000.0']
    for f in files:
        with open(f"results/road_A_100kph_30sec_30Hz/time_traces/results_w1_1_w2_{f}.pkl", 'rb') as file:
            state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax, par = pkl.load(
                file)

        paraWeight.append(float(f))
        paraComfort.append(rms(output_history[:, 0]))
        paraHolding.append(rms(output_history[:, 1]))
        paraComfortWeighted.append(wrms(output_history[:, 0], 1/(tValues[1]-tValues[0])))

    results = [paraWeight, paraComfort, paraHolding, paraComfortWeighted]

    with open('results/road_A_100kph_30sec_30Hz/results_weightSens.pkl', 'wb') as f:
        pkl.dump(results, f)

def regenerate_A2_results():
    paraWeight = []
    paraComfort = []
    paraComfortWeighted = []
    paraHolding = []

    files = ['0.01', 
             '26315.798947368417', 
             '52631.58789473684', 
             '78947.37684210525', 
             '105263.16578947367', 
             '131578.9547368421', 
             '157894.74368421052', 
             '184210.53263157894', 
             '210526.32157894736', 
             '236842.11052631578', 
             '263157.8994736842', 
             '289473.68842105265', 
             '315789.47736842104', 
             '342105.2663157894', 
             '368421.0552631579', 
             '394736.8442105263', 
             '421052.6331578947', 
             '447368.4221052631', 
             '473684.21105263155', 
             '500000.0']
    for f in files:
        with open(f"results/road_A_20kph_3sec_500Hz/time_traces/results_w1_1_w2_{f}.pkl", 'rb') as file:
            state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax, par = pkl.load(
                file)

        paraWeight.append(float(f))
        paraComfort.append(rms(output_history[:, 0]))
        paraHolding.append(rms(output_history[:, 1]))
        paraComfortWeighted.append(wrms(output_history[:, 0], 1/(tValues[1]-tValues[0])))

    results = [paraWeight, paraComfort, paraHolding, paraComfortWeighted]

    with open('results/road_A_20kph_3sec_500Hz/results_weightSens.pkl', 'wb') as f:
        pkl.dump(results, f)

def regenerate_bump_results():
    paraWeight = []
    paraComfort = []
    paraComfortWeighted = []
    paraHolding = []

    for i in np.linspace(1, 5e5, 20):
        with open(f"results/bump_20kph_10sec_500Hz/time_traces/results_w1_1_w2_{i}.pkl", 'rb') as file:
            state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax, par = pkl.load(
                file)

        paraWeight.append(float(i))
        paraComfort.append(rms(output_history[:, 0]))
        paraHolding.append(rms(output_history[:, 1]))
        paraComfortWeighted.append(wrms(output_history[:, 0], 1/(tValues[1]-tValues[0])))

    results = [paraWeight, paraComfort, paraHolding, paraComfortWeighted]

    with open('results/bump_20kph_10sec_500Hz/results_weightSens.pkl', 'wb') as f:
        pkl.dump(results, f)

def plot_sensitivity(name: str, plot=True):

    with open('results/' + name, 'rb') as f:
        paraWeight, paraComfort, paraHolding, paraComfortWeighted = pkl.load(f)

    plt.rcParams['axes.axisbelow'] = True
    plt.figure()
    plt.grid()
    plt.scatter(paraHolding, paraComfort, c=paraWeight, cmap='viridis')
    plt.ylabel("Comfort Index", fontsize=11)
    plt.xlabel("Road Holding Index", fontsize=11)
    cbar = plt.colorbar()
    cbar.set_label('Weight Parameter [-]')
    
    plt.figure()
    plt.grid()
    plt.scatter(paraHolding, paraComfortWeighted, c=paraWeight, cmap='viridis')
    plt.xlabel("Road Holding Index", fontsize=11)
    plt.ylabel("Weighted Comfort Index", fontsize=11)
    cbar = plt.colorbar()
    cbar.set_label('Weight Parameter [-]')
    
    if plot:
        plt.show()




if __name__ == "__main__":
    # plot_quarter("results_type_isoD_endT_120_f_30_tl_0.1_Np_10_quarter.pkl")

    # plot_half("results_type_isoD_endT_30_f_30_tl_0.1_Np_10_half.pkl")
    # plot_half("results_type_bump_endT_5_f_100_tl_0.1_Np_10_half.pkl")
    plot_different_np(["results_type_iso_D_endT_120_f_30_tl_2_Np_2_quarter.pkl",
                       "results_type_iso_D_endT_120_f_30_tl_2_Np_10_quarter.pkl",
                       "results_type_iso_D_endT_120_f_30_tl_2_Np_20_quarter.pkl"])
    # regenerate_A_results()
    # regenerate_D_results()
    # # regenerate_A2_results()
    # plot_sensitivity('road_D_25kph_30sec_30Hz/results_weightSens.pkl', plot=False)
    # plot_sensitivity('road_A_100kph_30sec_30Hz/results_weightSens.pkl', plot=False)
    # plot_sensitivity('bump_20kph_10sec_500Hz/results_weightSens.pkl', plot=False)
    # plt.show()
    # plot_sensitivity('road_A_20kph_3sec_500Hz/results_weightSens.pkl', plot=False)
    plt.show()
