% Weighted root mean square
% Inputs:
%   a_z - array with vertical acceleration values
%   ts - array with time of each acceleration value
% Output:
%   WRMS - weighted root mean square value of vertical acceleration time series
function [WRMS] = wrms(a_z, ts)
    s = tf('s');
    % Vertical acceleration weighting (ISO 2631-1)
    Wv = (87.72 * s^4 + 1138 * s^3 + 11336 * s^2 + 5452 * s + 5509) / ...
    (s^5 + 92.6854 * s^4 + 2549.83 * s^3 + 25969 * s^2 + 81057 * s + 79783);
    
    % Get the length of the time series
    n = length(a_z);
    
    % Compute the frequencies for the FFT
    frequencies = (0:n-1) * (1/n); % Frequency bins for FFT
    
    % Convert frequencies to angular frequencies
    omega = 2 * pi * frequencies;
    
    % Evaluate the transfer function at these frequencies
    H_f = freqresp(Wv, omega); % H_f is a frequency response data object
    
    % Since freqresp returns a 3D array, we need to reshape it to a 1D array
    H_f = squeeze(H_f);

    A_f = fft(a_z);

    % Define the weighting function in the frequency domain
    % n = length(acceleration);
    % frequencies = (0:n-1) * (1/n); % Frequency bins

    A_w_f = A_f.*H_f;
    
    disp(size(A_w_f))
    a_w = real(ifft(A_w_f));
    a_quad = a_w.^4;
    disp(size(a_quad))
    integral = sum((a_w.^4).*ts);
    disp(size(integral))
    WRMS = sqrt(1/(sum(ts))*integral);
end