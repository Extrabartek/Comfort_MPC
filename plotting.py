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

    active_wrms = wrms([], output_history[:, 0])
    passive_wrms = wrms([], output_pass_history[:, 0])
    quarter_wrms = wrms([], output_quarter_history[:, 0])

    print(f"Half wrms: {active_wrms}")
    print(f"Quarter wrms: {quarter_wrms}")
    print(f"Passive wrms: {passive_wrms}")

    print(f"The percentage improvement in WRMS is: {100 * (passive_wrms - active_wrms) / passive_wrms} %")

    active_wrms = wrmq(output_history[:, 0], [])
    passive_wrms = wrmq(output_pass_history[:, 0], [])
    quarter_wrms = wrmq(output_quarter_history[:, 0], [])

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

    active_wrms = wrms([], output_history[:, 0])
    passive_wrms = wrms([], output_pass_history[:, 0])

    print(f"The percentage improvement in WRMS is: {100 * (passive_wrms - active_wrms) / passive_wrms} %")

    active_wrms = wrmq(output_history[:, 0], [])
    passive_wrms = wrmq(output_pass_history[:, 0], [])

    print(f"The percentage improvement in WRMQ is {100 * (passive_wrms - active_wrms) / passive_wrms} %")
    plt.figure()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD: Body acceleration [(m/s^2)^2/Hz] ')
    freq_psd, result_psd = signal.periodogram(output_pass_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_active = signal.periodogram(output_history[:, 0].ravel(), fs=1 / (tValues[1] - tValues[0]))
    plt.loglog(freq_psd, result_psd, label='Passive PSD')
    plt.loglog(freq_psd, result_psd_active, label='Active PSD')
    freq_psd, result_psd = signal.periodogram(get_a_w(output_pass_history[:, 0].ravel())[0],
                                              fs=1 / (tValues[1] - tValues[0]))
    freq_psd, result_psd_active = signal.periodogram(get_a_w(output_history[:, 0].ravel())[0],
                                                     fs=1 / (tValues[1] - tValues[0]))
    plt.loglog(freq_psd, result_psd, label='Weighted Passive PSD')
    plt.loglog(freq_psd, result_psd_active, label='Weighted Active PSD')
    plt.legend()
    plt.xlim([0.2, 20])
    plt.hlines(1, 0, 1000, linestyle='--', colors='black')
    plt.ylim([1e-7, 5e0])
    plt.grid()
    plt.title('PSD of the body acceleration')


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

    plt.figure(figsize=(10, 6))
    plt.scatter(deflection_velocity, damping_force_history, label='Active Damper', color="#1192e8", alpha=0.75, s=15)
    plt.scatter(deflection_velocity_passive, damping_force_passive, label='Passive Damper', color="#da1e28", marker="D", alpha=0.95, s=15)
    # need to add parameters as a saved value
    z_values_range = np.linspace(np.min(deflection_velocity), np.max(deflection_velocity), num=len(deflection_velocity))
    plt.plot(z_values_range, z_values_range*csf, linestyle='-', color='#1c0f30', label='Nominal Damper', linewidth=1.5)
    plt.plot(z_values_range, z_values_range*csmax, linestyle='-.', color='#1c0f30', label='Active Damper Envelope', linewidth=1.5)
    plt.plot(z_values_range, z_values_range*csmin, linestyle='-.', color='#1c0f30')
    plt.xlabel('Suspension Velocity [m/s]', fontsize=16)
    plt.ylabel('Damping Force [N]', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.legend(fontsize=16)
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

def regenerate_D_results():
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
        with open(f"results/road_D_25kph_30sec_30Hz/time_traces/results_w1_1_w2_{f}.pkl", 'rb') as file:
            state_history, output_history, u_history, road_profile_front, road_profile_rear, damping_force_history, deflection_velocity, damping_force_passive, deflection_velocity_passive, tValues, state_pass_history, output_pass_history, csf, csr, csmin, csmax, par = pkl.load(
                file)

        paraWeight.append(float(f))
        paraComfort.append(rms(output_history[:, 0]))
        paraHolding.append(rms(output_history[:, 1]))
        paraComfortWeighted.append(wrms([], output_history[:, 0]))

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
        paraComfortWeighted.append(wrms([], output_history[:, 0]))

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
        paraComfortWeighted.append(wrms([], output_history[:, 0]))

    results = [paraWeight, paraComfort, paraHolding, paraComfortWeighted]

    with open('results/road_A_20kph_3sec_500Hz/results_weightSens.pkl', 'wb') as f:
        pkl.dump(results, f)

def plot_sensitivity(name: str, plot=True):

    with open('results/' + name, 'rb') as f:
        paraWeight, paraComfort, paraHolding, paraComfortWeighted = pkl.load(f)

    plt.rcParams['axes.axisbelow'] = True
    plt.figure()
    plt.grid()
    plt.scatter(paraHolding, paraComfort, c=paraWeight, cmap='viridis')
    plt.ylabel("Comfort Index", fontsize=12)
    plt.xlabel("Road Holding Index", fontsize=12)
    plt.colorbar()
    
    plt.figure()
    plt.grid()
    plt.scatter(paraHolding, paraComfortWeighted, c=paraWeight, cmap='viridis')
    plt.xlabel("Road Holding Index", fontsize=12)
    plt.ylabel("Weighted Comfort Index", fontsize=12)
    plt.colorbar()
    
    if plot:
        plt.show()




if __name__ == "__main__":
    plot_quarter("results_type_isoD_endT_120_f_30_tl_0.1_Np_10_quarter.pkl")

    # plot_half("results_type_isoD_endT_30_f_30_tl_0.1_Np_10_half.pkl")
    # plot_half("results_type_bump_endT_5_f_100_tl_0.1_Np_10_half.pkl")

    # regenerate_A_results()
    # regenerate_D_results()
    # regenerate_A2_results()
    # plot_sensitivity('bump_20kph_10sec_500Hz/results_weightSens.pkl', plot=False)
    # plot_sensitivity('road_A_100kph_30sec_30Hz/results_weightSens.pkl', plot=False)
    # plot_sensitivity('road_A_20kph_3sec_500Hz/results_weightSens.pkl', plot=False)
    plt.show()
