% Weighting factors for ride comfort assesment
% Using: http://resolver.tudelft.nl/uuid:3cfc846a-8f7d-4479-a88e-4615422071b6

clear, clc
s = tf('s');

n_points = 10e5;
f = linspace(0.01,1000,n_points);
w = f * 2 * pi;

%% Run section
%%% Weight filters HIGH ORDER %%%
% Vertical acceleration weighting (ISO 2631-1)
Wv = (87.72 * s^4 + 1138 * s^3 + 11336 * s^2 + 5452 * s + 5509) / ...
    (s^5 + 92.6854 * s^4 + 2549.83 * s^3 + 25969 * s^2 + 81057 * s + 79783);

% Horizontal acceleration weighting (ISO 2631-1)
Wh = (12.66 * s^3 + 163.7 * s^2 + 60.04 * s + 12.79) / ...
    (s^4 + 23.77 * s^3 + 236.1 * s^2 + 692.8 * s + 983.4);

% Motion sickness weighting (ISO 2631-1)
Wm = (0.1457 * s^4 + 0.2331 * s^3 + 13.75 * s^2 + 1.705 * s + 0.3596) / ...
    (s^5 + 7.757 * s^4 + 19.06 * s^3 + 28.37 * s^2 + 18.52 * s + 7.230);

figure(1);
bp1 = bodeplot(Wv);
setoptions(bp1,'FreqUnits','Hz', 'MagUnits', 'abs', 'FreqScale', 'log', 'MagScale', 'log');

figure(2);
bp1 = bodeplot(Wh);
setoptions(bp1,'FreqUnits','Hz', 'MagUnits', 'abs', 'FreqScale', 'log', 'MagScale', 'log');

figure(3);
bp1 = bodeplot(Wm);
setoptions(bp1,'FreqUnits','Hz', 'MagUnits', 'abs', 'FreqScale', 'log', 'MagScale', 'log');

[magnitudeWeightVertical, phaseWeightVertical] =  bode(Wv, w);
[magnitudeWeightHorizontal, phaseWeightHorizontal] =  bode(Wh, w);
[magnitudeWeightMotion, phaseWeightMotion] =  bode(Wm, w);

figure(4);
loglog(f, magnitudeWeightVertical(:))
hold on
grid on
grid minor
loglog(f, magnitudeWeightHorizontal(:))
loglog(f, magnitudeWeightMotion(:))
ylim([0.01, 2])
legend("Vertical","Horizontal","Motion")
xlabel("Frequency [Hz]")
ylabel("Gain [-]")

%% Run section
%%% Weight filters LOW ORDER %%%

Wv_low = (50 * s + 500) / (s^2 + 50 * s + 1200);
Wv_low = (86.51 * s + 546.1) / (s^2 + 82.17 * s + 1892);
Wh_low = (13.55 * s) / (s^2 + 12.90 * s + 47.16);
Wm_low = (0.8892 * s) / (s^2 + 0.8263 * s + 1.163);

[magnitudeWeightVertical_low, phaseWeightVertical_low] =  bode(Wv_low, w);
[magnitudeWeightHorizontal_low, phaseWeightHorizontal_low] =  bode(Wh_low, w);
[magnitudeWeightMotion_low, phaseWeightMotion_low] =  bode(Wm_low, w);

figure(5);
loglog(f, magnitudeWeightVertical_low(:))
hold on
grid on
grid minor
loglog(f, magnitudeWeightHorizontal_low(:))
loglog(f, magnitudeWeightMotion_low(:))
ylim([0.01, 2])
legend("Vertical","Horizontal","Motion")
xlabel("Frequency [Hz]")
ylabel("Gain [-]")

