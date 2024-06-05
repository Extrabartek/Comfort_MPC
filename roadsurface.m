%%% Various road platforms within matlab
% -> ISO (See provided_road_profile_generator.m)
% -> Standard bumb (https://ieeexplore.ieee.org/document/8357907)
% -> Bump sequence 
% -> Table top
% -> Sharp 
%
%
% Dimension site: https://www.dimensions.com/element/speed-bump-standard
%%%

clear, clc

% Time init
f = 100;            % Hz
tValues = 0:1/f:10; % [s] Array of to be evaluated timesteps

% Tunable parametrs (dependent on bump surface)
A = 0.3;            % m
V = 36 / 3.6;       % km/h
l = 30;             % m
L = 60;             % m

%Generate profile
profile = isolatedBump(tValues, A, V, l, L);

% Post processing
figure()
plot(tValues, profile, '-o')
ylim([0, ceil(A)])
ylabel('Height [m]')
xlabel('time [s]')


function heightProfile = isolatedBump(tValues, A, V, l, L)
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