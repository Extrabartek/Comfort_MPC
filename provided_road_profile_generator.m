%% Road profile generation based on ISO norm
rng(20);
k    = 3;                                   % Values For ISO Road A-B Roughness Classification, from 1 to 3 (to be checked)
V    = 40/3.6;                              % Vehicle Speed (m/s)
L    = 500;                                 % Length Of Road Profile (m
t    = L/V;                                 % measurement time (s)
f    = 100;                                % Sampling frequency (Hz)
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
plot(total_time,hx*1000);
xlim([0 max(total_time)]);
ylim([-50 50]);
xlabel('Time in s');
ylabel('Displacement in mm');
title('Road Profile');