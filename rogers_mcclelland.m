% Rogers and McClelland model 2008

% Authors: Hannah Sheahan and Andrew Saxe
% Date: 22/09/2020
% ----------------------------------------------------------------------- %
clear all;
close all;
clc;

colormap parula
FIGURE_DIR = 'figures/';

%% Setup inputs and attributes
n_contexts = 2;
n_items = 8;
n_domains = 3;
n_attributes = 15;  % coded output
n_outputs = n_attributes * n_domains * n_contexts;  % output nodes

% on each trial we have both one-hot context input and one-hot item input
% each item 1:8 will be assessed in each of 4 different contexts
max_itemsize = n_items*n_domains;
max_contextsize = n_contexts*n_domains;

% id/words for each input
all_domain_labels = {'company','mammals','monarchy'};
all_context_labels = {'human size','popularity','number of teeth', 'dietary preferences', 'friendliness', 'dog common ancestry'};
all_item_labels = { 'katie','michael','teresa','taika','Mr Perez','Ms Que','Mrs Wellie','Mr X', ...
                    'lion','elephant','monkey','mouse','sheep','ferret','gazelle','pig', ...
                    'chow chow','alaskan malamute','siberian husky','saluki','basenji','grey wolf','shar pei','akita inu' ...
                   };

% cycle through all possible inputs
count = 1;
lookup = zeros(n_domains, n_contexts, n_items);
for domain_idx = 1:n_domains
    for item_idx = 1:n_items
        for context_idx = 1:n_contexts    
            
            % set up one-hot input coding
            item_input(count,:) = int_to_onehot((domain_idx-1)*n_items + item_idx, max_itemsize);
            context_input(count,:) = int_to_onehot((domain_idx-1)*n_contexts +context_idx, max_contextsize);
            domains(count,:) = int_to_onehot(domain_idx,n_domains);
            
            % set up labels for each input
            context_labels(count,1) = all_context_labels(onehot_to_int(context_input(count,:)));
            domain_labels(count,1) = all_domain_labels(onehot_to_int(domains(count,:)));
            item_labels(count,1) = all_item_labels(onehot_to_int(item_input(count,:)));
            
            % set up integers for alternative lookup
            item_int(count) = item_idx;
            context_int(count) = context_idx;
            domain_int(count) = domain_idx;
            lookup(domain_idx,context_idx,item_idx) = count;
            
            count = count +1;
        end
    end
end
n_unique_inputs = size(item_input,1);

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
saveas(gcf,strcat(FIGURE_DIR,'Inputs_coding.png'));


%% Setup attributes

% each item has a set of attributes pertaining to it in each context, 
% and as with items and contexts, each domain has its own completely distinct set.

% the way this is coded is distributed across different units for each
% domain: its specified in Rumelhart and Todd 1993

% I think there are 14x4x4 attribute outputs in total, so 14 for each
% context in each domain, and each item activates some subset of these.

% construct item/attribute covariance matrix for context A (hierarchies)

context_cov_A = make_hierarchy(n_items, n_attributes);  % HRS note this is hacked together for now
context_cov_B = make_magnitude(n_items, n_attributes);
context_covs = permute(cat(3,context_cov_A,context_cov_B),[3,1,2]);

figure()
subplot(1,2,1)
imshow(context_cov_A)
title('hierarchy context')
xlabel('item #')
ylabel('attribute #')
subplot(1,2,2)
imshow(context_cov_B)
title('magnitude context')
xlabel('item #')
ylabel('attribute #')
saveas(gcf,strcat(FIGURE_DIR,'Item_attribute_covariance.png'));

% cycle through all possible outputs
attributes = zeros(n_unique_inputs, n_domains,n_contexts,n_attributes);

count = 1;
for domain_idx = 1:n_domains
    for item_idx = 1:n_items
        for context_idx = 1:n_contexts    
            
            attribute_ind = (domain_idx-1)*n_contexts +context_idx;
            context_cov = squeeze(context_covs(context_idx,:,:));
            attributes(count, domain_idx, context_idx, :) = squeeze(context_cov(:,item_idx)); % intentional spatial separation: each element in dim2+ is different node
                        
            count = count +1;
        end
    end
end


%% Plot the structure in the outputs/attributes for each context
% use Euclidean distance matrix, imshow of [items for all domains]x[items for all domains]
% in output space. This is like Fig R3A in Rogers/McClelland 2008

figure()
total_attribute_distance=zeros(n_contexts, n_domains*n_items, n_domains*n_items);
total_attribute_activity = zeros(n_contexts, n_domains*n_items, n_outputs);

for context_idx = 1:n_contexts
    attribute_activity = zeros(n_domains*n_items, n_outputs);
    for domain_idx = 1:n_domains
       for item_idx = 1:n_items
           count = lookup(domain_idx, context_idx, item_idx);
           item_attributes = reshape(attributes(count, :, :,:), 1, []);  % flatten across all output units
           attr_idx = (domain_idx-1)*n_items + item_idx;
           attribute_activity(attr_idx, :) = item_attributes;
       end 
    end

    % compute distance matrix over outputs
    D = pdist(attribute_activity);
    attribute_distance = squareform(D);
    attribute_distance(attribute_distance==eye(size(attribute_distance))) = 0;
    total_attribute_activity(context_idx,:,:) = attribute_activity;
    total_attribute_distance(context_idx,:,:) = attribute_distance;
    
    % rescale all values 0->1 for plotting
    attribute_activity = rescale(attribute_activity);
    attribute_distance = rescale(attribute_distance);

    subplot(2,2,(context_idx-1)*n_contexts + 1)
    imshow(attribute_activity)
    colorbar
    title('Output activity vectors')
    xlabel('output units')
    ylabel('items x domains')

    subplot(2,2,(context_idx-1)*n_contexts + 2)
    imshow(attribute_distance)
    colorbar
    title('Output RDM')
    xlabel('items x domains')
    ylabel('items x domains')
end
saveas(gcf,strcat(FIGURE_DIR,'Outputs_RDMs_by_context.png'));

% plot output RDM averaged/collapsed over the two contexts
mean_attr_dist = squeeze(mean(total_attribute_distance,1));
mean_attr_dist = rescale(mean_attr_dist);

figure()
imshow(mean_attr_dist,'InitialMagnification', 1000)
colorbar
title('Output RDM, collapsed across contexts')
xlabel('items x domains')
ylabel('items x domains')
saveas(gcf,strcat(FIGURE_DIR,'Outputs_RDM_contextcollapsed.png'));

%% Hierarchical cluster analysis of outputs (R3.B)
total_attribute_activity = squeeze(mean(total_attribute_activity,1));
Y = pdist(total_attribute_activity);
Z = linkage(Y);
figure()
dendrogram(Z);
title('Hierarchical cluster analysis of output activity')
xlabel('items x domains')
ylabel('Euclidean distance')
saveas(gcf,strcat(FIGURE_DIR,'Outputs_hierarchical_cluster_analysis.png'));


%%  Build the Rogers/McClelland network

% easier to do this in python with autograd...
clear vars count D domain_idx context_idx item_idx attr_idx attribute_ind Z Y
data=ws2struct();
save('mini_rogers_mclelland.mat', 'data')



