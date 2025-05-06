function [ prun_neuron ] = post_neurite_pruning( raw_matrix, neurite_ma, branch_node, leaf_node, prun_c )
%UNTITLED2 Summary of this function goes here
%   This function is used to prun the redundant neurites according to the
%   angle of each neurite.
%   [ prun_neuron ] = post_neurite_pruning( raw_matrix, neurite_ma, branch_node, leaf_node, prun_c )
%   raw_matrix: digitally reconstructed neuron 
%   neurite_ma: neurite matrix
%   branch_node: branch node
%   leaf_node: leaf node
%   prun_c: threshold angle, range from 0~pi.

    topo_node = union(branch_node, leaf_node);
    topo_ma = zeros(length(topo_node), 4);
    topo_ma(:, 4) = inf;
    topo_ma(:, 1) = topo_node;
    topo_con_ma = [];
    child_list_cell = cell(length(topo_node), 1);
    topo_con_cell = {};
    for i = 1:1:length(neurite_ma(:, 1)) %caculate the angle of each branch
         neu_1 = neurite_ma(i, :);
         neu_1(neu_1 == 0) = [];
         total_an = 0;
         for j = 1:1:length(neu_1) - 1
             neu_an_1 = raw_matrix(neu_1(j), 3:5) - raw_matrix(neu_1(j+1), 3:5);
             std_an_1 = raw_matrix(neu_1(j+1), 3:5) - raw_matrix(1, 3:5);
             an_1 = acos(sum(neu_an_1.* std_an_1)/(sqrt(sum(neu_an_1.^2))*sqrt(sum(std_an_1.^2))));
             if isnan(an_1)
                 an_1 = 0;
             end
             total_an = total_an + an_1;
         end
         av_an = total_an/(length(neu_1) - 1);
         ind = find(topo_ma(:, 1) == neu_1(1));
         if isempty(ind)
             disp('error! no topo node find!');
         end
         topo_ma(ind, 2) = length(neu_1) - 1;
         topo_ma(ind, 3) = av_an;
         topo_con_ma = [topo_con_ma ; [neu_1(end), neu_1(1)]];
         topo_con_cell = [topo_con_cell; {neu_1(1:end - 1)}];
    end
    ind_1 = ismember(topo_ma(:, 1), leaf_node);
    topo_ma(ind_1, 4) = topo_ma(ind_1, 3);
    %--------------------------------------------
    node_1 = leaf_node;
    while(~isempty(find(topo_ma(:, 4) == inf))) %caculate the angle of each branch according to its children.
        ind_1 = ismember(topo_con_ma(:, 2), node_1);
        par_node = topo_con_ma(ind_1, 1);
        par_node = union(par_node, par_node);
        for i = 1:1:length(par_node)
            ind_2 = find(ismember(topo_con_ma(:, 1), par_node(i)) == 1);
            child_1 = topo_con_ma(ind_2, 2);
            child_ind = find(ismember(topo_ma(:, 1), child_1) == 1);
            if isempty(find(topo_ma(child_ind, 4) == inf))
                par_ind = ismember(topo_ma(:, 1), par_node(i));
                topo_ma(par_ind, 4) = (topo_ma(par_ind, 3) .* topo_ma(par_ind, 2) + sum(topo_ma(child_ind, 4) .* topo_ma(child_ind, 2)))/(topo_ma(par_ind, 2) + sum(topo_ma(child_ind, 2)));
                topo_ma(par_ind, 2) = topo_ma(par_ind, 2) + sum(topo_ma(child_ind, 2));
                for j = 1:1:length(ind_2)
                    c_node_list = topo_con_cell{ind_2(j)};
                    child_list_cell{par_ind} = union(child_list_cell{par_ind}, c_node_list);
                    child_list_cell{par_ind} = union(child_list_cell{par_ind}, child_list_cell{child_ind(j)});
                end
                node_1(ismember(node_1, child_1)) = [];
                node_1 = [node_1; par_node(i)];
            end
        end
    end
    prun_ind = find(topo_ma(1:end, 4) > prun_c);
    prun_ind(prun_ind == 1) = [];
    prun_node = [];
    for i = 1:1:length(prun_ind)  %reconstruct the neuron
        node_2 = topo_node(prun_ind(i));
        if isempty(child_list_cell{prun_ind(i)}) && ~isempty(find(leaf_node == node_2))
            leaf_neurite = neurite_ma(find(neurite_ma(:, 1) == node_2), :);
            leaf_neurite(leaf_neurite == 0) = [];
            prun_node = union(prun_node, leaf_neurite(1:end - 1));
        else
            prun_node = union(prun_node, child_list_cell{prun_ind(i)});
        end
    end
    raw_matrix(prun_node, :)  = [];
    prun_neuron = tree_resort(raw_matrix);
    

end

