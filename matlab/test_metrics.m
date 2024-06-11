%% Test comfort metrics
x = 1:0.1:10;
acceleration_series = 0.1*exp(-x/10).*cos(2*pi*x) + (exp(-x)).*cos(10*2*pi*x);

% Float metrics
WRMS = wrms(acceleration_series, x);
WRMQ = wrmq(acceleration_series, x);
VDV = vdv(acceleration_series, x);

% Metric at each timestep

MTVV = mtvv(acceleration_series, x);
RWRMS = rwrms(acceleration_series, x);

%% plot
figure
hold on
plot(x, MTVV)
title("MTVV")
legend("MTVV")
hold off

figure
hold on
plot(x, acceleration_series)
title("Vertical Acceleration")
legend("Acceleration")
hold off

figure
hold on
plot(x, RWRMS)
title("Running weighted root mean square")
legend("RWRMS")
hold off

%% test raw

% a_z = acceleration_series;
% ts = x;
% 
% s = tf('s');
% Vertical acceleration weighting (ISO 2631-1)
% Wv = (87.72 * s^4 + 1138 * s^3 + 11336 * s^2 + 5452 * s + 5509) / ...
% (s^5 + 92.6854 * s^4 + 2549.83 * s^3 + 25969 * s^2 + 81057 * s + 79783);
% 
% Get the length of the time series
% n = length(a_z);
% 
% Compute the frequencies for the FFT
% frequencies = (0:n-1) * (1/n); % Frequency bins for FFT
% 
% Convert frequencies to angular frequencies
% omega = 2 * pi * frequencies;
% 
% Evaluate the transfer function at these frequencies
% [resp, freq] = freqresp(Wv, omega); % H_f is a frequency response data object
% resp = squeeze(resp)';
% 
% A_f = fft(a_z);
% 
% A_w_f = A_f.*resp;
