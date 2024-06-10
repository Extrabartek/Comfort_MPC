%% This is the file that will "run" the simulation,
%% generating the time trace of vertical acceleration

clear; close all; clc;

%% Constants
par.ms = 960;    % sprung mass of the vehicle chassis [kg]
par.I = 1222; % moment of inertia of the vehicle [kgm^2]
par.muf = 40;  % unsprung mass of the front axle [kg]
par.mur = 45;  % unsprung mass of the rear axle [kg]
par.ktf = 200000;    % stiffness of front tire material [N/m]
par.ktr = 200000;    % stiffness of rear tire material [N/m]
par.ksf = 18000;  % spring constant of the front axle [N/m]
par.ksr = 22000;  % spring constant of the rear axle [N/m]
par.csf = 1000;   % damping coefficient of the front axle [Ns/m]
par.csr = 1000;   % damping coefficient of the rear axle [Ns/m]
par.l1 = 1.3;    % front body length from the CG [m]
par.l2 = 1.5;    % rear body length from the CG [m]

par.a1 = 1/par.ms + (par.l1^2)/par.I;
par.a2 = 1/par.ms - (par.l1*par.l2)/par.I;
par.a3 = 1/par.ms + (par.l2^2)/par.I;


%% State Vector

state = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0];

% List of states:
% 1 - suspension deflection of the front car body
% 2 - vertical velocity of the front car body
% 3 - suspension deflection of the rear car body
% 4 - vertical velocity of the rear car body
% 5 - tire deflection of the front car body
% 6 - vertical velocity of the front wheel
% 7 - tire deflection of the rear car body
% 8 - vertical velocity of the rear wheel


%% Road Excitation

% Time init
f = 10000;            % Hz
tValues = 0:1/f:10; % [s] Array of to be evaluated timesteps

% Tunable parametrs (dependent on bump surface)
A = 0.9;            % m
V = 3.6 / 3.6;       % km/h
l = 5;             % m
L = 60;             % m

roadsurface;

rear_profile =   [zeros(1, ceil(f*(par.l1 + par.l2)/V)) profile];

%% The simulation loop
dt = 1/f;
n = length(tValues);
state_history = zeros(n, 8);
derivative_history = zeros(n, 8);
acceleration_history = zeros(n, 2);

d_skyhook = 0 * 1000.0;

for i = 1:1:n
    
    % calculate the forcing
    % the controller goes here
    
    
    %road(i,1) = sin(dt*i*1);
 
    %road(i,2) = sin(dt*i*100 - 0.01);

    derivative = half_car(state, [profile(i) rear_profile(i)], [0 0], par);
    
    state_history(i, :) = state;
    derivative_history(i, :) = derivative';

    state = (state' + derivative*dt)';
    acceleration_history(i, :) = acceleration_calc(state, [profile(i) rear_profile(i)], par);


end



%% Plotting
plot(tValues, profile, 'Color','red')
plot(tValues, rear_profile(1:n), 'Color', 'blue')