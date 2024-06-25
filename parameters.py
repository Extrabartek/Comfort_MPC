class Parameters:
    def __init__(self, sprung_mass, moment_of_inertia, unsprung_front, unsprung_rear, front_tire_stiffness,
                 rear_tire_stiffness, front_spring_constant, rear_spring_constant, front_damper_constant, 
                 rear_damper_constant, damper_min, damper_max,
                 front_body_length, rear_body_length):
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
        self.csmin = damper_min
        self.csmax = damper_max
        self.l1 = front_body_length
        self.l2 = rear_body_length
        self.a1 = 1 / self.ms + (self.l1 ** 2) / self.I
        self.a2 = 1 / self.ms - (self.l1 * self.l2) / self.I
        self.a3 = 1 / self.ms + (self.l2 ** 2) / self.I

par = Parameters(900, 1222, 75, 75, 200000, 200000, 27000, 27000, 2000, 2000, 500, 4500, 1.3, 1.5)