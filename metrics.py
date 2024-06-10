import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import freqresp
from scipy.signal import TransferFunction as tf
import matplotlib.pyplot as plt

def get_a_w(a_z):
    '''Get the weighted acceleration in the time domain using the frequency response of the weight filter'''
    # Define the frequency points
    n_points = len(a_z)
    f = np.linspace(0.01, 1000, n_points)
    w = f * 2 * np.pi

    # Define the transfer functions using tf
    Wv = tf([87.72, 1138, 11336, 5452, 5509], [1, 92.6854, 2549.83, 25969, 81057, 79783])

    # Calculate magnitude and phase
    _, magWeightVertical = freqresp(Wv, w)

    # Compute the FFT of the input acceleration
    A_f = fft(a_z)

    # Multiply the FFT of the input acceleration by the frequency response
    A_w_f = A_f * magWeightVertical

    # Inverse FFT to get weighted acceleration in time domain
    a_w = np.real(ifft(A_w_f))
    
    return a_w


def wrms(ts, a_z) -> np.float32:
    '''Weighted root mean square'''
    # Define the time step
    dt = ts[1] - ts[0]
    
    a_w = get_a_w(a_z)

    # Compute the integral for the WRMS calculation
    integral = np.sum((a_w ** 2) * ts)

    # Calculate WRMS
    WRMS = np.sqrt(1 / np.sum(ts) * integral)

    return WRMS

def wrmq(a_z, ts) -> np.float32:
    '''Weighted root mean quartic'''
    # Define the time step
    dt = ts[1] - ts[0]

    a_w = get_a_w(a_z)

    # Compute the integral for the WRMS calculation
    integral = np.sum((np.power(a_w,4)) * dt)

    # Calculate WRMS
    WRMQ = ((1 / np.sum(ts)) * integral)**(1/4)

    return WRMQ

def rwrms(a_z, ts) -> np.ndarray:
    '''Running weighted root mean square'''
    dt = ts[1] - ts[0]
    
    a_w = get_a_w(a_z)
    
    rwrms = np.zeros_like(a_w)
    
    rwrms[0] = np.sqrt((1/dt)*(a_z[0]**2)*dt)
    
    t_sum = np.cumsum(ts)
    
    for i in range(2, len(a_w)):
        rwrms[i] = np.sqrt((1/t_sum[i])*(t_sum[i-1]*rwrms[i-1]**2 + (a_w[i]**2)*dt))
        
    return rwrms
    
def vdv(a_z, ts) -> np.float32:
    '''Vibration dose value'''
    dt = ts[1] - ts[0]

    a_w = get_a_w(a_z)
    
    vdv = (np.sum(np.power(a_w, 4) * dt))**(1/4)
    
    return vdv

def mtvv(rwrms_vals) -> np.ndarray:
    '''Maximum transient vibration value'''
    mtvv = np.zeros_like(rwrms_vals)
    for i in range(1, len(rwrms_vals)):
        mtvv[i] = np.max(rwrms_vals[:i])
    return mtvv

if __name__ == "__main__":
    START_TIME = 0
    END_TIME = 10
    FREQ = 100 # Hz
    dt = 1 / FREQ
    ts = np.linspace(START_TIME, END_TIME, int((END_TIME - START_TIME) / dt))
    num_samples = len(ts)
    print(f"Number of samples: {num_samples}")
    
    a_z = np.sin(2 * np.pi * ts) + np.random.normal(0, 0.1, len(ts))
    
    # plot the input acceleration
    
    plt.figure()
    plt.plot(ts, a_z)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.title('Input Acceleration')
    plt.grid(True)
    plt.show()
    
    # print the float metrics
    
    print(f"WRMS: {wrms(ts, a_z)}")
    print(f"WRMQ: {wrmq(a_z, ts)}")
    print(f"VDV: {vdv(a_z, ts)}")
    
    # plot the running metrics
    
    rwrms_vals = rwrms(a_z, ts)
    
    plt.figure()
    plt.plot(ts, rwrms_vals)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.title('Running WRMS')
    plt.grid(True)
    plt.show()
    
    # plot the maximum transient vibration value
    
    mtvv_vals = mtvv(rwrms_vals)
    
    plt.figure()
    plt.plot(ts, mtvv_vals)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.title('Maximum Transient Vibration Value')
    plt.grid(True)
    plt.show()
    
    # plot the weighted acceleration metrics