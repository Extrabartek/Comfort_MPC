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

state = [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1];

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

% this is where the road profile will be generated

T = 100000;

uf = zeros(T, 1);

% the road profile needs to be delayed for the rear wheels

ur = zeros(T,1);

road = [uf ur];

%% The simulation loop

n = length(uf); % number of simulation steps
state_history = zeros(n, 8);
derivative_history = zeros(n, 8); 

dt = 0.0001;



for i = 1:1:T
    
    % calculate the forcing
    % the controller goes here

    
    %road(i,1) = sin(dt*i*1);
 
    %road(i,2) = sin(dt*i*100 - 0.01);

    derivative = half_car(state, [0 0], [0 0], par);
    
    state_history(i, :) = state;
    derivative_history(i, :) = derivative';

    state = (state' + derivative*dt)';


end

plot((1:1:T), state_history(:,1))
