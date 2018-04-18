clear;close all;

% n is the number of samples
% d is the dimensionality of each sample
[A_sample, y] = iris_dataset;
A_sample = A_sample';
%y = y';
y1=vec2ind(y)';
[n,d] = size(A_sample);
num_block=2; %number of blocks
%% This section is used to combine the clustering results of all blocks into a uniform representation.
U_column_index = sparse([]);
num_raw_clusters = 0;
alpha = 0.015;
A_block = cell(1:num_block);
y_block = cell(1:num_block);

X_opt_optimal = zeros(n/num_block,num_block);
num_opt_clusters = zeros(num_block,1);
id = 1:n;
for bi = 1:num_block
    A_temp = A_sample(mod(id,num_block)==bi-1,:);% uniform sampling, i.e., put samples into different blocks in order
    A_block{bi} = A_temp; % store arrays with different dimensions in struct
    y_temp = y1(mod(id,num_block)==bi-1,:);
    y_block{bi} = y_temp;% store labels to compute nmi later
    [lambda_opt_primal, block_opt_primal] = slim_convex_clustering_dual(A_temp,alpha);
    X_opt_optimal(:,bi) = block_opt_primal;
    rng('default');  % For reproducibility
    eva = evalclusters(block_opt_primal,'kmeans','gap','KList',[1:6])
    %num_opt_clusters(:,bi) = eva.CriterionValues;
    [C,I] = max(eva.CriterionValues);
    num_opt_clusters(bi) = I;
    [IDX,C,sumd] = kmeans(block_opt_primal,num_opt_clusters(bi));
    nmi(bi) = MutualInfo(IDX,y_temp);
    [cluster_set,ia2,u_column_index] = unique(IDX,'stable');
    
    u_column_index = u_column_index + num_raw_clusters;
    U_column_index = [U_column_index; u_column_index];
    % vertically stack u_i, i.e., to get raw clustering memberships
    num_raw_clusters = num_raw_clusters + length(cluster_set);
end
U_row_index = 1:n;
U_value_index = ones(n,1);
U = sparse(U_row_index,U_column_index,U_value_index);

alpha_refine = 0.0105;
gamma_refine = 0.0008;
% A_blocks = [];
% y_blocks = [];
% for i=1:num_block
%     A_blocks = [A_blocks; A_block{i}];
%     y_blocks = [y_blocks; y_block{i}];
% end
UU = sparse([]);
for bi = 1:num_block
    UU(mod(id,num_block)==bi-1,:) = U(1:length(A_block{bi}),:);
end
[ S ] = clusters_refinement( A_sample, UU, alpha_refine, gamma_refine);
rng('default');  % For reproducibility
eva = evalclusters(S,'kmeans','gap','KList',[1:6])
[C,I] = max(eva.CriterionValues);
num_opt_clusters = I;
[IDX,C,sumd] = kmeans(S,num_opt_clusters);
Nmi = MutualInfo(IDX,y1);
%% Evaluate the performance
% NMI
A = [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3];
B = [1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3];
%B = [2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 3, 2, 2, 3, 3, 3];
MIhat = MutualInfo(A,B);
