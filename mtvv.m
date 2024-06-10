% Maximum transient vibration value
% Inputs:
%   a_z - array with vertical acceleration values
%   ts - array with time of each acceleration value
% Output:
%   MTVV - weighted root mean square value of vertical acceleration time series
function MTVV = mtvv(a_z, ts)
    MTVV = zeros(1,length(a_z));
    RWRMS = rwrms(a_z, ts);
    for i = 1:length(ts)
        MTVV(1, i) = max(RWRMS(1:i));
    end
end