function [cov] = make_magnitude(n_items,n_attributes)
% Produce a magnitude covariance matrix that shows how attributes can
% encode the similarity relations between items
% Date: 18/11/2020
% ----------------------------------------------------------------------- %

cov = zeros(n_attributes, n_items);
j=floor(n_items/2);
stop_ind = flip([1:n_items]);   

count = 1;
for ind = 2:n_items
    j = stop_ind(ind);
    cov(count,end-j+1:end) = ones(1,j);
    
    cov(end-count,1:j) = ones(1,j);
    count = count + 1;
    
end

