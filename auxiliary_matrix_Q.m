function [ Q ] = auxiliary_matrix_Q( n )
%AUXILIARY_MATRIX_Q Summary of this function goes here
%   Detailed explanation goes here

row_Q = nchoosek(n,2); % the number of rows of Q
Q = zeros(row_Q,n);    % initialize Q with all entries 0
j=2;  % the index of entry -1 in Q
j1=1; % the index of entry  1 in Q
%% construct the matrix Q which is n(n-1)/2 in dimensionality
for i=1:row_Q 
    Q(i,j1)=1;        
    Q(i,j)=-1;
    j=j+1;
    if(j>n)
        j1=j1+1;
        j=j1+1;
    end
end

end

