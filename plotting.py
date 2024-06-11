import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import freqresp
from scipy.signal import TransferFunction as tf
import matplotlib.pyplot as plt
from metrics import get_a_w, rms

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