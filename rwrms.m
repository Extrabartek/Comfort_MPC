function [RWRMS] = rwrms(a_z, ts)
    RWRMS = zeros(length(a_z));
    for i = 1:length(ts)
        t = ts(i);
        RWRMS(i) = wrms(a_z(t<=ts), ts(t<=ts));
    end
end