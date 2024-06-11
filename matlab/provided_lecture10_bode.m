% B.Shyrokau
% b.shyrokau@tudelft.nl
% Intelligent Vehicles & Cognitive Robotics
% Department of Cognitive Robotics
% Faculty of Mechanical, Maritime and Materials Engineering
% Delft University of Technology, The Netherlands
% 2022

clc; clear; close all;


%% quater car parameters
m_s = 410;          % sprung mass, kg
m_u = 45;           % unsprung mass, kg
k_t = 230000;       % tire vertical stiffness, N/m
k_s = 25500;        % suspension vertical stiffness, N/m
% natural frequency
wn_s = sqrt(k_s*k_t / (k_s + k_t)/m_s);
% critical damping ratio
c_c = 2 * m_s * wn_s;
% damping ratio
d_s = 0.3 * c_c;
% sprung mass natural frequency
fn_s = sqrt(k_s*k_t / (k_s + k_t)/m_s)/ 2/pi;
% unsprung mass natural frequency
fn_u = sqrt((k_s + k_t)/m_u)/ 2/pi;

% preparation for bode
n_points = 5000;
w = linspace(0.1,100,n_points)*2*pi;

% 1 - mass ratio
% 2 - stiffness ratio
% 3 - damping
testtype = 2;

%% Outputs of C_ss
% az1
% az2
% Fdyn
% z1
% z2
% dz

% Effect of mass ratio
if testtype == 1
    mass_ratio = [0.05 0.10 0.20];
    for i = 1:length(mass_ratio)
        m_u = mass_ratio(i) * m_s;
        A_ss = [-d_s/m_s  d_s/m_s  -k_s/m_s  k_s/m_s; ...
            d_s/m_u -d_s/m_u   k_s/m_u  -(k_s + k_t)/m_u; ...
            1 0 0 0;
            0 1 0 0];
        B_ss = [0; k_t/m_u; 0; 0];
        C_ss = [-d_s/m_s  d_s/m_s  -k_s/m_s  k_s/m_s; d_s/m_u -d_s/m_u   k_s/m_u  -(k_s + k_t)/m_u; 0 0 0 -k_t; 0 0 1 0; 0 0 0 1; 0 0 -1 1;];
        D_ss = [0; k_t/m_u; k_t; 0; 0; 0];
        sys = ss(A_ss,B_ss,C_ss,D_ss);
        systf = tf(sys);
        % Bode plots
        [az1(i,:),az1_phase(i,:)] = bode(systf(1,1),w);
        [az2(i,:),az2_phase(i,:)] = bode(systf(2,1),w);
        [Fdyn(i,:),Fdyn_phase(i,:)] = bode(systf(3,1),w);
        [z1(i,:),z1_phase(i,:)] = bode(systf(4,1),w);
        [z2(i,:),dz_phase(i,:)] = bode(systf(5,1),w);
        [dz(i,:),dz_phase(i,:)] = bode(systf(6,1),w);
    end
end

% Effect of stiffness ratio
if testtype == 2
    stiffness_ratio = [5 8 10];
    for i = 1:length(stiffness_ratio)
        k_s = k_t / stiffness_ratio(i);
        A_ss = [-d_s/m_s  d_s/m_s  -k_s/m_s  k_s/m_s; ...
            d_s/m_u -d_s/m_u   k_s/m_u  -(k_s + k_t)/m_u; ...
            1 0 0 0;
            0 1 0 0];
        B_ss = [0; k_t/m_u; 0; 0];
        C_ss = [-d_s/m_s  d_s/m_s  -k_s/m_s  k_s/m_s; d_s/m_u -d_s/m_u   k_s/m_u  -(k_s + k_t)/m_u; 0 0 0 -k_t; 0 0 1 0; 0 0 0 1; 0 0 -1 1;];
        D_ss = [0; k_t/m_u; k_t; 0; 0; 0];
        sys = ss(A_ss,B_ss,C_ss,D_ss);
        systf = tf(sys);
        % Bode plots
        [az1(i,:),az1_phase(i,:)] = bode(systf(1,1),w);
        [az2(i,:),az2_phase(i,:)] = bode(systf(2,1),w);
        [Fdyn(i,:),Fdyn_phase(i,:)] = bode(systf(3,1),w);
        [z1(i,:),z1_phase(i,:)] = bode(systf(4,1),w);
        [z2(i,:),dz_phase(i,:)] = bode(systf(5,1),w);
        [dz(i,:),dz_phase(i,:)] = bode(systf(6,1),w);
    end
end

% Effect of damping ratio
if testtype == 3
    damping_ratio = [0.1 0.3 0.707];
    for i = 1:length(damping_ratio)
        d_s = damping_ratio(i) * c_c;
        A_ss = [-d_s/m_s  d_s/m_s  -k_s/m_s  k_s/m_s; ...
            d_s/m_u -d_s/m_u   k_s/m_u  -(k_s + k_t)/m_u; ...
            1 0 0 0;
            0 1 0 0];
        B_ss = [0; k_t/m_u; 0; 0];
        C_ss = [-d_s/m_s  d_s/m_s  -k_s/m_s  k_s/m_s; d_s/m_u -d_s/m_u   k_s/m_u  -(k_s + k_t)/m_u; 0 0 0 -k_t; 0 0 1 0; 0 0 0 1; 0 0 -1 1;];
        D_ss = [0; k_t/m_u; k_t; 0; 0; 0];
        sys = ss(A_ss,B_ss,C_ss,D_ss);
        systf = tf(sys);
        % Bode plots
        [az1(i,:),az1_phase(i,:)] = bode(systf(1,1),w);
        [az2(i,:),az2_phase(i,:)] = bode(systf(2,1),w);
        [Fdyn(i,:),Fdyn_phase(i,:)] = bode(systf(3,1),w);
        [z1(i,:),z1_phase(i,:)] = bode(systf(4,1),w);
        [z2(i,:),dz_phase(i,:)] = bode(systf(5,1),w);
        [dz(i,:),dz_phase(i,:)] = bode(systf(6,1),w);
    end
end

%% Processing
sFont = 14;

% Transmissibility Ratio
figure();
set(gcf,'Color','white');
loglog(w/(2*pi),az1,'LineWidth',1.5)
hold all
xlim([0.1 100])
% ylim([1e-4 10])
grid on
xlabel('Frequency, Hz','FontSize',sFont,'FontWeight','normal');
ylabel('Transmissibility Ratio','FontSize',sFont,'FontWeight','normal');
set(gca,'FontSize',sFont)
grid on;
switch testtype
    case 1
        legend(strcat('mass ratio = ', num2str(mass_ratio')))
    case 2
        legend(strcat('stiffness ratio = ', num2str(stiffness_ratio')))
    case 3
        legend(strcat('damping ratio = ', num2str(damping_ratio')))
end

% Suspension Travel Ratio
figure();
set(gcf,'Color','white');
loglog(w/(2*pi),dz,'LineWidth',1.5)
hold all
xlim([0.1 100])
ylim([1e-2 10])
grid on
xlabel('Frequency, Hz','FontSize',sFont,'FontWeight','normal');
ylabel('Suspension Travel Ratio','FontSize',sFont,'FontWeight','normal');
set(gca,'FontSize',sFont)
grid on;
switch testtype
    case 1
        legend(strcat('mass ratio = ', num2str(mass_ratio')))
    case 2
        legend(strcat('stiffness ratio = ', num2str(stiffness_ratio')))
    case 3
        legend(strcat('damping ratio = ', num2str(damping_ratio')))
end

% Dynamic Tire Deflection Ratio
figure();
set(gcf,'Color','white');
loglog(w/(2*pi),Fdyn / k_t,'LineWidth',1.5)
hold all
xlim([0.1 100])
ylim([5e-4 10])
grid on
xlabel('Frequency, Hz','FontSize',sFont,'FontWeight','normal');
ylabel('Dynamic Tire Deflection Ratio','FontSize',sFont,'FontWeight','normal');
set(gca,'FontSize',sFont)
grid on;
switch testtype
    case 1
        legend(strcat('mass ratio = ', num2str(mass_ratio')))
    case 2
        legend(strcat('stiffness ratio = ', num2str(stiffness_ratio')))
    case 3
        legend(strcat('damping ratio = ', num2str(damping_ratio')))
end
