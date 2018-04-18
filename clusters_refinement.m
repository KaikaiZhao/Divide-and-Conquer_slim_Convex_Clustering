function [ S ] = clusters_refinement( A_blocks, U, alpha_refine, gamma_refine)
%CLUSTERS_REFINEMENT Summary of this function goes here
%   Detailed explanation goes here
%   alpha_refine is a hyperparameter to control 
%   gamma_refine is a hyperparameter to ensure that N is sparse

%% The dual problem formulation during 'conquer' phase is as follows,
% min F'G'GF - 2/d(F'G'A1)
%  the corresponding code x = quadprog(H,f,A,b,Aeq,beq,lb,ub)
%  the formal formulation is min 1/2*(x'*H*x) + f'*x
%  such that,
%           Ax <= b
%           Aeq*x = beq
%           lb <= x <= ub
[n,d] = size(A_blocks);
Q = auxiliary_matrix_Q(n);
row_Q = size(Q,1);
blockA1 = [eye(row_Q); -eye(row_Q)];
blockA2 = [eye(n); -eye(n)];
A_conquer = blkdiag(blockA1, blockA2);

b_conquer = [alpha_refine*ones(2*row_Q,1); gamma_refine*ones(2*n,1)];
K = size(U,2);% the number of raw clusters
Aeq_conquer = [zeros(row_Q,K); U]';
beq_conquer = zeros(1, K)';
G = [Q' eye(n)];
H = 2*(G'*G);
f_conquer = (-2/d)*G'*A_blocks*ones(d,1);
F = quadprog(H,f_conquer,A_conquer,b_conquer,Aeq_conquer,beq_conquer);
S = 1/d*A_blocks*ones(d,1) - G*F;

end

