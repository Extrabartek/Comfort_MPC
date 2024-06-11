%% Run Simulation
close all;

%states
%1 zs
%2 zs dot
%3 zu
%4 zu dot



% Constraint matrix
Aineq = E_tilde + F_tilde*B_tilde;

% Hessian cost representation
H = B_tilde.'*Q_tilde*B_tilde + R_tilde;

% Time vector
t = 0:N_steps-1;

% Initialize arrays to store results
x_results = zeros(2, N_steps);
u_results = zeros(1, N_steps);

% Initialize state
xk = x0;
uk = 0;
% Quadprog options
options = optimoptions('quadprog', 'Display', 'off');
warning('off', 'all');

for i = 1:N_steps

    % Store results
    x_results(:, i) = xk;
    u_results(i) = uk;

    % Contraints RHS vector
    bineq = G_tilde - F_tilde*A_tilde*xk;

    % State cost
    f = 2*xk.'*A_tilde.'*Q_tilde*B_tilde;

    % Get next optimal input
    u_pred = quadprog(H*2,f,Aineq,bineq,[],[],[],[],[],options);
    if size(u_pred, 1) == 0
        disp("Infeasible optimization problem at timestep " + i);
        break
    end

    % Update control input
    uk = u_pred(1);

    % Get next state
    xk = A*xk + B*uk;
end

warning('on', 'all');

% Store final value results
x_results(:, end) = xk;
u_results(end) = uk;
