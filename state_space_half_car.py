import numpy as np


class Parameters:
    def __init__(self, sprung_mass, moment_of_inertia, unsprung_front, unsprung_rear, front_tire_stiffness,
                 rear_tire_stiffness, front_spring_constant, rear_spring_constant, front_damper_constant,
                 rear_damper_constant, front_body_length, rear_body_length):
        self.ms = sprung_mass
        self.I = moment_of_inertia
        self.muf = unsprung_front
        self.mur = unsprung_rear
        self.ktf = front_tire_stiffness
        self.ktr = rear_tire_stiffness
        self.ksf = front_spring_constant
        self.ksr = rear_spring_constant
        self.csf = front_damper_constant
        self.csr = rear_damper_constant
        self.l1 = front_body_length
        self.l2 = rear_body_length
        self.a1 = 1 / self.ms + (self.l1 ** 2) / self.I
        self.a2 = 1 / self.ms + (self.l1 * self.l2) / self.I
        self.a3 = 1 / self.ms + (self.l2 ** 2) / self.I


def half_car_state_space(par: Parameters):
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
    F = np.array([[0, par.a1, 0, par.a2, 0, -1 / par.muf, 0, 0],
                  [0, par.a2, 0, par.a3, 0, 0, 0, -1 / par.mur]]).T

    Q = np.array([[1, 0],
                  [0, 1]])

    C1_dash = np.array([[1 / par.ms * -par.ksf, 1 / par.ms * -par.csf, 1 / par.ms * -par.ksr, 1 / par.ms * -par.csr, 0,
                         1 / par.ms * par.csf, 0, 1 / par.ms * par.csr],
                        [1 / par.I * par.l1 * par.ksf, 1 / par.I * par.l1 * par.csf, 1 / par.I * -par.l2 * par.ksr,
                         1 / par.I * -par.l2 * par.csr, 0, 1 / par.I * -par.l1 * par.csf, 0,
                         1 / par.I * par.l2 * par.csr]])

    C = np.dot(Q, C1_dash)
    E = np.dot(Q, np.array([[1 / par.ms, 1 / par.ms],
                            [-par.l1 / par.I, par.l2 / par.I]]))

    return [A, B, F, C, E]


if __name__ == "__main__":
    pass
