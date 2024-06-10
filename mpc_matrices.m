function [A_tilde,B_tilde,E_tilde,F_tilde,G_tilde,Q_tilde,R_tilde] = mpc_matrices(Np,A,B,E,F,G,Q,R)
%MPC_MATRICES Summary of this function goes here
%   Detailed explanation goes here
%Initialize B tilde matrix
B_tilde = [];
for i = 1:Np
    Bi = [];
    for j = 1:Np
        if i-j < 0
            Bij = zeros(size(B));
        else
            Bij = A^(i-j) * B;
        end
        Bi = [Bi Bij];
    end
    B_tilde = [B_tilde; Bi];
end

%Initialize the other tilde matrices
A_tilde = [];
G_tilde = [];
R_tilde = [];
Q_tilde = [];
E_tilde = [];
F_tilde = [];
for i = 1:Np
    E_tilde = blkdiag(E_tilde, E);
    F_tilde = blkdiag(F_tilde, F);
    G_tilde = [G_tilde; G];
    A_tilde = [A_tilde; A^i];
    R_tilde = blkdiag(R_tilde, R);
    Q_tilde = blkdiag(Q_tilde, Q);
end
end

