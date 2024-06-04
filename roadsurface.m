%%% Various road platforms within matlab
% -> ISO (See provided_road_profile_generator.m)
% -> Standard bumb (https://ieeexplore.ieee.org/document/8357907)
% -> Bump sequence 
% -> Table top
% -> Sharp 

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
l = 10;             % m
L = 10;             % m

%Generate profile
profile = isolatedBump(tValues, A, V, l, L);

% Post processing
figure()
plot(tValues, profile, '-o')
ylim([0, ceil(A)])
ylabel('Height [m]')
xlabel('time [s]')


function heightProfile = isolatedBump(tValues, A, V, l, L)
    
    heightProfile = [];
    
    for t = tValues
        
        if (l / V) < t && t < (l + L) / V
            outHeight = (A / 2) * (1 - cos((2*pi*V*t)/L));
        else
            outHeight = 0;
        end
        
        heightProfile = [heightProfile, outHeight];
    end
end