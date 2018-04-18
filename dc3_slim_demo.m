%% This code is used to test slim_convex_clustering_dual function
clear;
clc;
close all;

% n is the number of samples
% d is the dimensionality of each sample
%[A_sample, y] = load('./datasets/iris.mat');
[A_sample, y] = iris_dataset;
A_sample = A_sample';
%y = y';
y1=vec2ind(y);
[n,d] = size(A_sample);
i=1;
alpha=0.016:0.001:0.04;
len_alpha=length(alpha);
%lambda_primal=zeros(len_alpha,1);
%X_primal=zeros(len_alpha,1);
nmi=zeros(len_alpha,1);
for i=1:len_alpha
    [lambda_opt_primal, X_opt_primal] = slim_convex_clustering_dual(A_sample,alpha(i));
    lambda_primal(i,:)=lambda_opt_primal;
    X_primal(i,:)=X_opt_primal;
    rng('default');  % For reproducibility
    eva = evalclusters(X_primal(i,:)','kmeans','gap','KList',[1:6])
    num_opt_clusters(i) = eva.OptimalK;
    [IDX,C,sumd] = kmeans(X_opt_primal,num_opt_clusters(i));
    nmi(i) = MutualInfo(IDX,y1);
end
%save('divide_hyperPara_results.mat','alpha','lambda_primal','X_primal','nmi');
