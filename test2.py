from state_space_half_car import half_car_state_space, Parameters
import scipy.signal as signal

par = Parameters(960, 1222, 40, 45, 200000,
                    200000, 18000, 22000, 1000,
                    1000, 1.3, 1.5)

A, B, F, C, E = half_car_state_space(par)

ss = signal.cont2discrete((A, B, C, E), 0.01)

A1 = ss[0]
B1 = ss[1]
C1 = ss[2]
E1 = ss[3]

print(A)
print(A1)
print(B)
print(B1)
print(C)
print(C1)
print(E)
print(E1)