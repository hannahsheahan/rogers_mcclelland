function [one_hot] = int_to_onehot(int_input,max_size)
% Convert integer input to one-hot encoding
% Author: Hannah Sheahan
% Date: 22/09/2020

one_hot = zeros(1,max_size);
one_hot(int_input) = 1;
end

