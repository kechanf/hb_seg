function [neurite_matrix] = split_neurite_delta(branch_node_ids, leaf_node_ids, parent_list, child_list)
%This function is used to split the neurites from a neuron or neuron cluster.
 
neurite_id = 1;
% always include root node conceptually as a branch node, even if
% it has only one stem
if isempty(find(branch_node_ids ==1))
    branch_node_ids = [1;branch_node_ids];
end
while ~isempty(leaf_node_ids)
    node_id = 1;
    leaf_id= leaf_node_ids(1);
    parent_id = leaf_id;
    while isempty(find(branch_node_ids == parent_id))
        leaf_id = parent_id;
        Neu_matrix(neurite_id,node_id) = leaf_id;
        % find parent
        parent_id = parent_list(parent_list(:,1)==leaf_id,2);
        child_list((child_list(:,2) == leaf_id),:) = [];
        node_id = node_id + 1;
    end
    Neu_matrix(neurite_id,node_id) = parent_id;
    child_list((child_list(:,2) == leaf_id ),:) = [];
    [l_m l_n] = size(leaf_node_ids);
    if isempty(find(child_list(:,1) == parent_id))&&(parent_id~=branch_node_ids(1))
        leaf_node_ids = [leaf_node_ids; parent_id];
        branch_node_ids((branch_node_ids == parent_id)) = [];
    end
    leaf_node_ids(1) = [];
    neurite_id = neurite_id + 1;
end
Neu_matrix
neurite_matrix = Neu_matrix;