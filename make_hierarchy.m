function [cov] = make_hierarchy(n_items,n_attributes)
% Produce a hierarchical covariance matrix that shows how attributes can
% encode the similarity relations between items
% HRS just going to hack this together for now totally hardcoded
% Date: 18/11/2020
% ----------------------------------------------------------------------- %

cov = zeros(n_attributes, n_items);
i=1;
j=floor(n_items/2);

start_ind = [1,1,5,1,3,5,7,1,2,3,4,5,6,7,8];  % HRS hack
stop_ind = [8,4,8,2,4,6,8,1,2,3,4,5,6,7,8];   % HRS hack
count = 1;
for ind = 1:n_attributes
    i = start_ind(ind);
    j = stop_ind(ind);
    cov(count,i:j) = ones(1,j-i+1);
    count = count + 1;
end

