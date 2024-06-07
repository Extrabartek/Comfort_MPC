% Vibration dose value
% Inputs:
%   a_z - array with vertical acceleration values
%   ts - array with time of each acceleration value
% Output:
%   VDV - Vibration dose value of vertical acceleration time series
function VDV = vdv(a_z, ts)
    % Vertical acceleration weighting (ISO 2631-1)
    Wv = (87.72 * s^4 + 1138 * s^3 + 11336 * s^2 + 5452 * s + 5509) / ...
    (s^5 + 92.6854 * s^4 + 2549.83 * s^3 + 25969 * s^2 + 81057 * s + 79783);

    A_f = fft(a_z);

    % Define the weighting function in the frequency domain
    % n = length(acceleration);
    % frequencies = (0:n-1) * (1/n); % Frequency bins

    A_w_f = A_f.*Wv;
    
    a_w = real(ifft(A_w_f));
        
    VDV = (sum((a_w.^4)).*ts)^(1/4);
end