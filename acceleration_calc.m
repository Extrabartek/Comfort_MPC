function accelerations = acceleration_calc(states, road, par)
%ACCELERATION_CALC Calculates acceleration based on current state and road
%   profile
accelerations = zeros(1, 2);

q_1 = 1; % weighting of linear vertical acc
q_2 = 1; % weighting of roational acc

Q = [[q_1  0];
      [0    q_2]];

C1_dash = [1/par.ms * [-par.ksf -par.csf -par.ksr -par.csr 0 par.csf 0 par.csr];
           1/par.I *  [par.l1*par.ksf par.l1*par.csf -par.l2*par.ksr -par.l2*par.csr 0 -par.l1*par.csf 0 par.l2*par.csr];];

C = Q*C1_dash;
E = Q * [[1/par.ms 1/par.ms];[-par.l1/par.I par.l2/par.I];];

accelerations = C*states' + E*road';

end

