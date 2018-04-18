function [ lambda, X ] = slim_convex_clustering_dual(A_block, alpha)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%   alpha is a hyperparameter which is used to control the number of clusters
[n,d] = size(A_block);
Q = auxiliary_matrix_Q(n);
row_Q = size(Q,1);

%% solve the quadratic program for the 'divide' phase
%  the corresponding code x = quadprog(H,f,A,b,Aeq,beq,lb,ub)
%  the formal formulation is min 1/2*(x'*H*x) + f'*x
% such that,
%           Ax <= b
%           Aeq*x = beq
%           lb <= x <= ub
% Our dual problem is min lamdba' *Q*Q'* lambda - 4*lambda'*Q*A_sample*1
% to adapt to the form of the quadprog function, reformulate our dual
% problem as follows,
% min 1/2*lamdba'*(2*Q*Q')*lambda - lambda'*(4*Q*A_sample*1)
f1 = ones(d, 1);
H = 2*(Q*Q');
f = -4*Q*A_block*f1;
E_lambda = eye(row_Q);
A = [E_lambda; -E_lambda];

b = alpha*ones(row_Q*2, 1);
lambda = quadprog(H,f,A,b); % the solution to the dual problem
% We obtain the solution to the primal problem via the relationships between X
% and lambda. The specific code is as follows,
X = 1/d*(A_block*ones(d,1) - 1/2*Q'*lambda);

end

