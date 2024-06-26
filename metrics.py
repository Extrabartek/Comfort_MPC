import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import freqresp
from scipy.signal import TransferFunction as tf
import matplotlib.pyplot as plt

def rms(a_z) -> np.float32:
    '''Root mean square'''
    # Calculate the integral for the RMS calculation
    RMS = np.sqrt(np.mean(a_z**2))
    return RMS

def rmq(a_z) -> np.float32:
    '''Root mean quartic'''
    # Calculate the integral for the RMQ calculation
    RMQ = (np.mean(a_z**4))**(1/4)
    return RMQ


def get_a_w(a_z, fs):
    '''Get the weighted acceleration in the time domain using the frequency response of the weight filter'''
    # Define the frequency points
    a_z = a_z.ravel()
    n_points = len(a_z)

    # Compute the FFT of the input acceleration
    A_f = fft(a_z)
    f = fftfreq(n_points, d=1/fs)

    w = f * 2 * np.pi

    # Define the transfer functions using tf
    Wv = tf([87.72, 1138, 11336, 5452, 5509], [1, 92.6854, 2549.83, 25969, 81057, 79783])

    # Calculate magnitude and phase
    w, magWeightVertical = freqresp(Wv, w)

    # Multiply the FFT of the input acceleration by the frequency response
    A_w_f = A_f * magWeightVertical

    # Inverse FFT to get weighted acceleration in time domain
    a_w = np.real(ifft(A_w_f))
    
    return a_w, w


def wrms(a_z, fs) -> np.float32:
    '''Weighted root mean square'''    
    
    a_w,_ = get_a_w(a_z, fs)
    # Compute the integral for the WRMS calculation
    WRMS = rms(a_w)

    return WRMS

def wrmq(a_z, fs) -> np.float32:
    '''Weighted root mean quartic'''
    # Define the time step
    a_w,_ = get_a_w(a_z, fs)
    WRMQ = rmq(a_w)

    return WRMQ

def rwrms(a_z, ts) -> np.ndarray:
    '''Running weighted root mean square'''
    dt = ts[1] - ts[0]
    
    a_w,_ = get_a_w(a_z, 1/dt)
    
    rwrms = np.zeros_like(a_w)
    
    rwrms[0] = np.sqrt((a_z[0]**2))
    
    t_sum = np.cumsum(ts)
    
    for i in range(2, len(a_w)):
        
        # T = t_sum[i]
        # T_prev = t_sum[i-1]
        # rwrms[i] = (1/np.sqrt(T)) * np.sqrt(T_prev * rwrms[i-1]**2 + a_w[i]**2 * dt)
        rwrms[i] = wrms(a_z[:i], 1/dt)
        
    return rwrms
    
def vdv(a_z, ts) -> np.float32:
    '''Vibration dose value'''
    dt = ts[1] - ts[0]

    a_w,_ = get_a_w(a_z, 1/dt)
    
    vdv = (np.sum(np.power(a_w, 4))*dt)**(1/4)
    
    return vdv

def mtvv(rwrms_vals) -> np.ndarray:
    '''Maximum transient vibration value'''
    mtvv = np.zeros_like(rwrms_vals)
    for i in range(1, len(rwrms_vals)):
        mtvv[i] = np.max(rwrms_vals[:i])
    return mtvv

if __name__ == "__main__":
    START_TIME = 0
    END_TIME = 0.25 # s
    SAMPLING_FREQ = 1000 # Hz
    dt = 1 / SAMPLING_FREQ
    ts = np.linspace(START_TIME, END_TIME, int((END_TIME - START_TIME) / dt))
    num_samples = len(ts)
    print(f"Number of samples: {num_samples}")
    g = 9.81 # m/s^2
    w = 40 # rad/s
    
    a_z = np.exp(-w*ts)*(0.3*g)*np.sin(2*np.pi*w*ts)
        
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