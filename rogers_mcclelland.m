% Rogers and McClelland model 2008

% Authors: Hannah Sheahan and Andrew Saxe
% Date: 22/09/2020
% ----------------------------------------------------------------------- %
clear all;
close all;

%% Setup inputs and attributes
n_contexts = 4;
n_items = 8;
n_domains = 4;
n_attributes = 14;  % outputs

% on each trial we have both one-hot context input and one-hot item input
% each item 1:8 will be assessed in each of 4 different contexts
max_itemsize = n_items*n_domains;
max_contextsize = n_contexts*n_domains;

% cycle through all possible inputs
count = 1;
for domain_idx = 1:n_domains
    for item_idx = 1:n_items
        for context_idx = 1:n_contexts      
            item_input(count,:) = int_to_onehot((domain_idx-1)*n_items + item_idx, max_itemsize);
            context_input(count,:) = int_to_onehot((domain_idx-1)*n_contexts +context_idx, max_contextsize);
            domains(count,:) = int_to_onehot(domain_idx,n_domains);
            count = count +1;
        end
    end
end

% Check that unique inputs appropriately made
figure()
subplot(1,3,1)
imshow(item_input)
axis equal;
title('Items')
ylabel('input id #')

subplot(1,3,2)
imshow(context_input)
axis equal;
title('Contexts')
ylabel('input id #')

subplot(1,3,3)
imshow(domains)
axis equal;
title('Domains')
ylabel('input id #')

% Looks good so far.
%% Setup attributes

% each item has a set of attributes pertaining to it in each context, 
% and as with items and contexts, each domain has its own completely distinct set.

% the way this is coded is distributed across different units for each
% domain: its specified in Rumelhart and Todd 1993

% I think there are 14x4x4 attribute outputs in total, so 14 for each
% context in each domain, and each item activates some subset of these.





