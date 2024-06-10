### Various road platforms within matlab
# -> ISO (See provided_road_profile_generator.m)
# -> Standard bumb (https://ieeexplore.ieee.org/document/8357907)
# -> Bump sequence 
# -> Table top
# -> Sharp (TODO)
# -> Circle (TODO)
# -> Use a combination to get a complete test setup
# -> Long sinusoidal road (IDEA)
#
# Dimension site: https://www.dimensions.com/element/speed-bump-standard
###

import numpy as np
import matplotlib.pyplot as plt


def isolatedBump(f, A, V, l, L, endTime) -> list[float]:
    # Generate an isolated bump, based on the specified input criteria.
    #
    # Inputs:
    # -------
    # tValues = 1D array of the time values at which the road surface should 
    # be created.
    # A       = Amplitude of the bump in meter
    # V       = Velocity of the vehicle in meter per second
    # l       = Start position of the bump in meter
    # L       = Length of the bump in meter
    # Outputs:
    # -------
    # heightProfile = 1D array containing the height profile with values 
    #                   provided in meter [0,0,0,0,0.1,0.3,...]
    
    tValues = np.arange(0, endTime + 1/f, 1/f); # [s] Array of to be evaluated timestamps
    heightProfile = np.zeros(len(tValues))
    
    for idx in range(len(tValues)):
        
        t = tValues[idx]
        
        if ((l / V) <= t) and (t <= (l + L) / V):
            height = (A / 2) * (1 - np.cos(2 * np.pi * (V * t - l) / L))
        else:
            height = 0
        
        heightProfile[idx] = height

    return heightProfile


def isolatedTable(f, V, l, endTime):
    # Generate an isolated table, based on the specified input criteria.
    # Sources:
    # https://www.dimensions.com/element/speed-bump-table
    # https://highways.dot.gov/safety/speed-management/traffic-calming-eprimer/module-3-part-2#3.12
    # https://nacto.org/publication/urban-street-design-guide/street-design-elements/vertical-speed-control-elements/speed-table/
    #
    # Inputs:
    # -------
    # tValues = 1D array of the time values at which the road surface should 
    # be created.
    # V       = Velocity of the vehicle in meter per second
    # l       = Start position of the table in meter
    # Outputs:
    # -------
    # heightProfile = 1D array containing the height profile with values 
    #                   provided in meter [0,0,0,0,0.1,0.3,...]
    
    tValues = np.arange(0,endTime+1/f,1/f) # [s] Array of to be evaluated timestamps
    heightProfile = np.zeros(len(tValues))
    
    A = 0.09                   # 9cm table in meter
    L = 6.4                    # 6.4m length of table in meter
    slope = 1 / 25             # slope of the ramp
    tableLength = A / slope    # distance before/ after the table

    for idx in range(len(tValues)):
        
        t = tValues[idx]

        if ((l / V) <= t) and (t <= (l + L) / V):
            if ((l + tableLength) / V) >= t:
                height = slope * (t * V - l)
            elif ((l + L - tableLength) / V) <= t:
                height = -slope * (t * V - (l + L))
            else:
                height = A
        else:
            height = 0
        
        heightProfile[idx] = height
    
    return heightProfile


def isoRoad(f, V, t) -> None:
    # Road profile generation based on ISO norm
    np.random.seed(20)
    
    k = 3   # Values For ISO Road A-B Roughness Classification, from 1 to 3 (to be checked)
    # V    = 40/3.6 # Vehicle Speed (m/s)
    # L    = 500    # Length Of Road Profile (m)
    L    = t * V    # measurement time (s)
    # t    = L/V    # measurement time (s)
    # f    = 100    # Sampling frequency (Hz)
    N = f * t    # Number of data points
    B = L / N    # Sampling Interval (m)
    dn = 1 / L    # Frequency Band
    n0 = 0.1      # Spatial Frequency (cycles/m)
    # n = dn : dn : N*dn   # Spatial Frequency Band
    n = np.arange(dn, N*dn + dn, dn)     
    phi  = 2*np.pi*np.random.rand(len(n))                  # Random Phase Angle
    Amp1 = np.sqrt(dn)*(2**k)*(1e-3)*(n0/n) # Amplitude for Road  Class A-B
    # x = 0:B:L;    # Abscissa Variable from 0 to L
    x = np.arange(0,L+B, B) # Abscissa Variable from 0 to L
    hx = np.zeros(len(x))
    for i in range(len(x)):
        hx[i] = sum(Amp1*np.cos(2*np.pi*n*x[i] + phi))

    # total_time = linspace(0,t,length(hx));

    heightProfile = hx
    
    return heightProfile

if __name__ == "__main__":

    ## Time init
    f = 100            # Hz

    ## Tunable parameters (dependent on bump surface)
    A = 0.3            # m
    V = 36 / 3.6       # km/h
    l = 10             # m
    L = 10             # m

    ## Generate profile
    profileBump = isolatedBump(f, A, V, l, L, 10)
    profileTable = isolatedTable(f, V, l, 10)
    profileISO = isoRoad(f, V, 10)

    profileTime = np.arange(0, 10 + 1/f, 1/f)
    # profile = [profileBump, profileTable];
    # profileTime = [tValues, tValues + tValues(end)];

    # Post processing

    plt.plot(profileTime, profileTable, '-o')
    # ylim([0, ceil(A)])
    plt.ylabel('Height [m]')
    plt.xlabel('time [s]')
    plt.show()

    # profile = isoRoad(f, V, L);
    # tValues = linspace(0,L/V,length(profile));
    # 
    # figure()
    # plot(tValues, profile);
    # xlabel('Time in s');
    # ylabel('Displacement in mm');
    # title('Road Profile');