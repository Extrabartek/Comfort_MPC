% Running weighted root mean square
% Inputs:
%   a_z - array with vertical acceleration values
%   ts - array with time of each acceleration value
% Output:
%   RWRMS - array of running weighted root mean square value of vertical acceleration time series
function [RWRMS] = rwrms(a_z, ts)
    RWRMS = zeros(length(a_z));
    for i = 1:length(ts)
        t = ts(i);
        RWRMS(i) = wrms(a_z(t<=ts), ts(t<=ts));
    end
end