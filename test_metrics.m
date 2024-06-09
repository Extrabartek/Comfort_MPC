%% Test comfort metrics
x = 1:0.001:10;
acceleration_series = cos(x);

% Float metrics
WRMS = wrms(acceleration_series, x);
% WRMQ = wrmq(acceleration_series, x);
% VDV = vdv(acceleration_series, x);
% 
% % Metric at each timestep
% 
% MTVV = mtvv(acceleration_series, x);
% RWRMS = rwrms(acceleration_series, x);
% 
% fprinft("WRMS: %.4f", WRMS)
% fprintf("WRMQ: %.4f", WRMQ)
% fprintf("VDV:  %.4f", VDV)
% 
% figure;
% subplot(3,1,1);
% hold on
% plot(x, acceleration_series)
% plot(x, MTVV)
% plot(x, RWRMS)
% title("Time varying metrics")
% legend("Acceleration", "MTVV", "RWRMS")
% hold off

