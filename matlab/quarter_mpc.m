function [outputArg1,outputArg2] = untitled(state,par)
%UNTITLED Summary of this function goes here
%   state
%       1 zu-zr
%       2 zu'
%       3 zs-zu
%       4 zs'
%   input
%       1 f
%       2 zr

Af = [0                1                0               0;
      -par.ktf/par.muf -par.csf/par.muf par.ksf/par.muf par.csf/par.muf;
      0                -1               0               1;
      0                par.csf/par.ms   -par.ksf/par.ms -par.csf/par.ms];

Bf = [0              0;
      par.ms/par.muf par.ktf/par.muf;
      0              0;
      -1             0];

Ef = [];

Ff
Ab = [0                1                0               0;
      -par.ktb/par.mub -par.csb/par.mub par.ksb/par.mub par.csb/par.mub;
      0                -1               0               1;
      0                par.csb/par.ms   -par.ksb/par.ms -par.csb/par.ms];

Bb = [0              0;
      par.ms/par.mub par.ktb/par.mub;
      0              0;
      -1             0];


end

