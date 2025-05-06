function [ new_matrix ] = soma_leaf_prun( raw_matrix, child_list, radius_c )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    elem_A = raw_matrix(:,1);
    [count_elem, node_ind] = hist(child_list(:,1),elem_A);
    branch_1 = node_ind(count_elem > 2);
    del_node = [];
    for i = 1:1:length(branch_1)
        node_1 = child_list(child_list(:, 1) == branch_1(i), 2);
        radius = raw_matrix(branch_1(i), 6) * radius_c;
        for j = 1:1:length(node_1)
            c_node = node_1(j);
            del_node_1 = c_node;
            while(~isempty(child_list(child_list(:, 1) == c_node, 2)))
                c_node = child_list(child_list(:, 1) == c_node, 2);
                dist = pdist2(raw_matrix(branch_1(i), 3:5), raw_matrix(c_node, 3:5));
                if length(c_node) > 1 || dist > radius 
                    del_node_1 = [];
                    break;
                else
                    del_node_1 = [del_node_1; c_node];
                end
            end
            del_node = [del_node ; del_node_1];
        end
    end
    raw_matrix(del_node, :) = [];
    new_matrix = tree_resort(raw_matrix);
            


end

