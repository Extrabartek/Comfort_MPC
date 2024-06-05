% LQR suspension control
% this script is adopted from 
% Ulsoy, A. G., Peng, H., & Çakmakci, M. (2012). Automotive control systems. Cambridge University Press.
%
% B.Shyrokau
% RO47017 Vehicle Dynamics & Control, 2022
% Use and distribution of this material outside the RO47017 course 
% only with the permission of the course coordinator
clc; clear all; close all;
% QC parameters
m_s = 410;                                           % sprung mass, kg
m_u = 45;                                            % unsprung mass, kg
k_t = 230000;                                        % tire vertical stiffness, N/m
k_s = 25500;                                         % suspension vertical stiffness, N/m
wn_s = sqrt(k_s*k_t / (k_s + k_t)/m_s);              % sprung mass natural frequency
d_s = 0.3 * (2 * m_s * wn_s);                        % damping ratio
% frequency range determination
n_points = 5000;
w = linspace(0.1,100,n_points)*2*pi;

%% 
% passive QC
% state matrix
A = [0, 1, 0, 0;...                                  % unsprung displacement - road disturbance
    -k_t / m_u, -d_s / m_u, k_s / m_u, d_s / m_u;... % unsprung mass acceleration
    0, -1, 0, 1;...                                  % sprung displacement - unsprung displacement
    0, d_s / m_s, -k_s / m_s, -d_s / m_s];           % sprung mass acceleration

B = [0, m_s / m_u, 0, -1]';                          % control matrix
G = [0, k_t / m_u, 0, 0]';

% output matrix
C = [k_t 0 0 0;...                                   % tire force
     0 0 1 0;...                                     % suspension stroke
     A(4,:)];                                        % sprung mass acceleration
Dw = [-k_t; 0; 0];
Du = [0; 0; -1];                                     % feedthrough matrix
% Weights selection for use in performance index
r1 = 4e+4;      % comfort weight
r2 = 5e+3;      % road holding weight
r3 = 0;         % control effort weight
Rxx = A(4,:)'*A(4,:) + diag([r1 0 r2 0]);
Rxu = -A(4,:)';
Ruu = 1 + r3;
% LQR optimal gain
[Kr,~] = lqr(A,B,Rxx,Ruu,Rxu);
Ac = (A - B  * Kr);
Cc = (C - Du * Kr);
% Calculate bode response: passive & active
[mag_p_tire_force, phase_p_tire_force] = bode(A,G,C(1,:),Dw(1),1,w);
[mag_a_tire_force, phase_a_tire_force] = bode(Ac,G,Cc(1,:),Dw(1),1,w);
[mag_p_susp, phase_p_susp] = bode(A,G,C(2,:),Dw(2),1,w);
[mag_a_susp, phase_a_susp] = bode(Ac,G,Cc(2,:),Dw(2),1,w);
[mag_p_ride, phase_p_ride] = bode(A,G,C(3,:),Dw(3),1,w);
[mag_a_ride, phase_a_ride] = bode(Ac,G,Cc(3,:),Dw(3),1,w);

%% Processing
sFont = 14;
hFig = figure();
set(hFig, 'Position', [100 100 1400 600])
subplot(1,3,1)
set(gcf,'Color','white');
loglog(w/(2*pi),mag_p_ride,'LineWidth',1.5);
hold all
loglog(w/(2*pi),mag_a_ride,'LineWidth',1.5);
xlabel('Frequency, Hz','FontSize',sFont,'FontWeight','normal');
ylabel('Sprung mass acceleration','FontSize',sFont,'FontWeight','normal');
set(gca,'FontSize',sFont)
grid on;
legend('passive', 'LQR');
subplot(1,3,2)
set(gcf,'Color','white');
loglog(w/(2*pi),mag_p_susp,'LineWidth',1.5);
hold all
loglog(w/(2*pi),mag_a_susp,'LineWidth',1.5);
xlabel('Frequency, Hz','FontSize',sFont,'FontWeight','normal');
ylabel('Suspension travel','FontSize',sFont,'FontWeight','normal');
set(gca,'FontSize',sFont)
grid on;
subplot(1,3,3)
set(gcf,'Color','white');
loglog(w/(2*pi),mag_p_tire_force,'LineWidth',1.5);
hold all
loglog(w/(2*pi),mag_a_tire_force,'LineWidth',1.5);
xlabel('Frequency, Hz','FontSize',sFont,'FontWeight','normal');
ylabel('Dynamic Tire Deflection Ratio','FontSize',sFont,'FontWeight','normal');
set(gca,'FontSize',sFont)
grid on;