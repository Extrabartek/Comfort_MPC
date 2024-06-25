import numpy as np
from parameters import Parameters


def half_car_state_space(par: Parameters):
    # List of states:
    # 1 - suspension deflection of the front car body
    # 2 - vertical velocity of the front car body
    # 3 - suspension deflection of the rear car body
    # 4 - vertical velocity of the rear car body
    # 5 - tire deflection of the front car body
    # 6 - vertical velocity of the front wheel
    # 7 - tire deflection of the rear car body
    # 8 - vertical velocity of the rear wheel
    A = np.array([[0, 1, 0, 0, 0, -1, 0, 0],
                  [-par.ksf * par.a1, -par.csf * par.a1, -par.ksr * par.a2, -par.csr * par.a2, 0, par.csf * par.a1, 0,
                   par.csr * par.a2],
                  [0, 0, 0, 1, 0, 0, 0, -1],
                  [-par.ksf * par.a2, -par.csf * par.a2, -par.ksr * par.a3, -par.csr * par.a3, 0, par.csf * par.a2, 0,
                   par.csr * par.a3],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [par.ksf / par.muf, par.csf / par.muf, 0, 0, -par.ktf / par.muf, -par.csf / par.muf, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, par.ksr / par.mur, par.csr / par.mur, 0, 0, -par.ktr / par.mur, -par.csr / par.mur]])
    B = np.array([[0, 0, 0, 0, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -1, 0]]).T
    F = np.array([[0, -par.a1, 0, -par.a2, 0, 1 / par.muf, 0, 0],
                  [0,-par.a2, 0,-par.a3, 0, 0, 0, 1 / par.mur]]).T

    # Output
    # 1. Sprung mass acceleration
    # 2. Pitch acceleration
    C = np.array([[1 / par.ms * -par.ksf, 1 / par.ms * -par.csf, 1 / par.ms * -par.ksr, 1 / par.ms * -par.csr, 0,
                         1 / par.ms * par.csf, 0, 1 / par.ms * par.csr],
                  [1 / par.I * par.l1 * par.ksf, 1 / par.I * par.l1 * par.csf, 1 / par.I * -par.l2 * par.ksr,
                         1 / par.I * -par.l2 * par.csr, 0, 1 / par.I * -par.l1 * par.csf, 0,
                         1 / par.I * par.l2 * par.csr]])

    D = np.array([[0, 0, -1 / par.ms, -1 / par.ms],
                  [0, 0, par.l1 / par.I, -par.l2 / par.I]])


    return [A, B, F, C, D]


if __name__ == "__main__":
    pass
