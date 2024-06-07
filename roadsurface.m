%%% Various road platforms within matlab
% -> ISO (See provided_road_profile_generator.m)
% -> Standard bumb (https://ieeexplore.ieee.org/document/8357907)
% -> Bump sequence 
% -> Table top
% -> Sharp (TODO)
% -> Circle (TODO)
% -> Use a combination to get a complete test setup
% -> Long sinusoidal road (IDEA)
%
% Dimension site: https://www.dimensions.com/element/speed-bump-standard
%%%

% Time init
f = 100;            % Hz
tValues = 0:1/f:10; % [s] Array of to be evaluated timesteps

% Tunable parametrs (dependent on bump surface)
A = 0.3;            % m
V = 36 / 3.6;       % km/h
l = 10;             % m
L = 10;             % m

%Generate profile
profileBump = isolatedBump(tValues, A, V, l, L);
profileTable = isolatedTable(tValues, V, l);
[profileISO, timeISO] = isoRoad(f, V, L);

% profile = [profileBump, profileTable];
% profileTime = [tValues, tValues + tValues(end)];

% Post processing
figure()
plot(timeISO, profileISO, '-o')
% ylim([0, ceil(A)])
ylabel('Height [m]')
xlabel('time [s]')

% profile = isoRoad(f, V, L);
% tValues = linspace(0,L/V,length(profile));
% 
% figure()
% plot(tValues, profile);
% xlabel('Time in s');
% ylabel('Displacement in mm');
% title('Road Profile');


function [heightProfile, time] = isolatedBump(tValues, A, V, l, L)
    % Generate an isolated bump, based on the specified input criteria.
    %
    % Inputs:
    % -------
    % tValues = 1D array of the time values at which the road surface should 
    % be created.
    % A       = Amplitude of the bump in meter
    % V       = Velocity of the vehicle in meter per second
    % l       = Start position of the bump in meter
    % L       = Length of the bump in meter
    % Outputs:
    % -------
    % heightProfile = 1D array containing the height profile with values 
    %                   provided in meter [0,0,0,0,0.1,0.3,...]

    heightProfile = zeros(1,length(tValues));
    
    for idx = 1:length(tValues)
        
        t = tValues(idx);
        
        if ((l / V) <= t) && (t <= (l + L) / V)
            height = (A / 2) * (1 - cos(2 * pi * (V * t - l) / L));
        else
            height = 0;
        end
        
        heightProfile(idx) = height;
    
    end
end


function [heightProfile, time] = isolatedTable(tValues, V, l)
    % Generate an isolated table, based on the specified input criteria.
    % Sources:
    % https://www.dimensions.com/element/speed-bump-table
    % https://highways.dot.gov/safety/speed-management/traffic-calming-eprimer/module-3-part-2#3.12
    % https://nacto.org/publication/urban-street-design-guide/street-design-elements/vertical-speed-control-elements/speed-table/
    %
    % Inputs:
    % -------
    % tValues = 1D array of the time values at which the road surface should 
    % be created.
    % V       = Velocity of the vehicle in meter per second
    % l       = Start position of the table in meter
    % Outputs:
    % -------
    % heightProfile = 1D array containing the height profile with values 
    %                   provided in meter [0,0,0,0,0.1,0.3,...]
    
    heightProfile = zeros(1,length(tValues));
    
    A = 0.09;                   % 9cm table in meter
    L = 6.4;                    % 6.4m length of table in meter
    slope = 1 / 25;             % slope of the ramp
    tableLength = A / slope;    % distance before/ after the table

    for idx = 1:length(tValues)
        
        t = tValues(idx);

        if ((l / V) <= t) && (t <= (l + L) / V)
            if ((l + tableLength) / V) >= t
                height = slope * (t * V - l);
            elseif ((l + L - tableLength) / V) <= t
                height = -slope * (t * V - (l + L));
            else
                height = A;
            end
        else
            height = 0;
        end
        
        heightProfile(idx) = height;
    
    end
end


function [heightProfile, total_time] = isoRoad(f, V, L)
    % Road profile generation based on ISO norm
    rng(20);
    k    = 3;                                   % Values For ISO Road A-B Roughness Classification, from 1 to 3 (to be checked)
    % V    = 40/3.6;                            % Vehicle Speed (m/s)
    % L    = 500;                               % Length Of Road Profile (m)
    t    = L/V;                                 % measurement time (s)
    % f    = 100;                               % Sampling frequency (Hz)
    N    = f*t;                                 % Number of data points
    B    = L/N ;                                % Sampling Interval (m)
    dn   = 1/L;                                 % Frequency Band
    n0   = 0.1;                                 % Spatial Frequency (cycles/m)
    n    = dn : dn : N*dn;                      % Spatial Frequency Band
    phi  = 2*pi*rand(size(n));                  % Random Phase Angle
    Amp1 = sqrt(dn)*(2^k)*(1e-3)*(n0./n);       % Amplitude for Road  Class A-B
    x = 0:B:L-B;                                % Abscissa Variable from 0 to L
    hx = zeros(size(x));
    for i=1:length(x)
        hx(i) = sum(Amp1.*cos(2*pi*n*x(i)+ phi));
    end
    total_time = linspace(0,t,length(hx));

    heightProfile = hx;
end