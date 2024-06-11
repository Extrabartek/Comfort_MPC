from gurobipy import Model, GRB, MVar
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import matplotlib.pyplot as plt
from state_space_half_car import Parameters

model = Model('test')

d = model.addVar(vtype=GRB.BINARY, name='delta')
x1 = model.addVar(vtype=GRB.CONTINUOUS, name='x1', lb=1.999999999, ub = 3.000000001)
x2 = model.addVar(vtype=GRB.CONTINUOUS, name='x2', lb=2.999999999, ub = 4.000000001)
model.update()
model.addGenConstrIndicator(d, 1, x1 - x2, GRB.GREATER_EQUAL, 0)
model.update()
model.setObjective(x1 + x2, GRB.MINIMIZE)
model.update()
model.write('model.lp')

model.optimize()

print(x1.x, x2.x, d.x)