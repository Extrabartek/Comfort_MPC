% Weighted root mean square
% Inputs:
%   a_z - array with vertical acceleration values
%   ts - array with time of each acceleration value
% Output:
%   WRMS - weighted root mean square value of vertical acceleration time series
function [WRMS] = wrms(a_z, ts)
    % Vertical acceleration weighting (ISO 2631-1)
    Wv = (87.72 * s^4 + 1138 * s^3 + 11336 * s^2 + 5452 * s + 5509) / ...
    (s^5 + 92.6854 * s^4 + 2549.83 * s^3 + 25969 * s^2 + 81057 * s + 79783);

    A_f = fft(a_z);

    % Define the weighting function in the frequency domain
    % n = length(acceleration);
    % frequencies = (0:n-1) * (1/n); % Frequency bins

    A_w_f = A_f.*Wv;
    
    a_w = real(ifft(A_w_f));

    WRMS = sqrt((1/sum(ts))*(sum((a_w.^2)).*ts));
end