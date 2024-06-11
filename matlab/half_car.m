function derivatives = half_car(states, road, damper_force, par)
%HALF_CAR Calculates derivatives using half-car model
%   Adding a single step in time to calculate state derivatives using the
%   half-car model, with the addition of an active damper.

derivatives = zeros(size(states));

A = [            [0  1   0   0   0   -1   0   0];
                 [-par.ksf*par.a1 -par.csf*par.a1 -par.ksr*par.a2 -par.csr*par.a2   0   par.csf*par.a1  0   par.csr*par.a2];
                 [0  0   0   1   0   0   0   -1];
                 [-par.ksf*par.a2  -par.csf*par.a2  -par.ksr*par.a3   -par.csr*par.a3  0  par.csf*par.a2 0   par.csr*par.a3];
                 [0  0   0   0   0   1   0   0];
     1/par.muf * [par.ksf par.csf  0  0  -(par.ktf)    -(par.csf)    0   0];
                 [0  0   0   0   0   0   0   1];
     1/par.mur * [0  0   par.ksr par.csr    0   0   -par.ktr    -par.csr];
     ];

B = [[0 0   0   0   -1  0   0   0];
     [0 0   0   0   0   0   -1  0]]';

F = [[0 par.a1  0   par.a2  0   -1/par.muf  0   0];
     [0 par.a2  0   par.a3  0   0   0   -1/par.mur];]';


derivatives = A*states' + B*road' + F*damper_force';


end

