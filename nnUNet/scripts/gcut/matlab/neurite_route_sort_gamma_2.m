function [fitness_matrix, connect_ind, del_neurite_ma ] = neurite_route_sort_gamma_2(Neurite_matrix, connect_ma, Neuron, total_fitness_ma, soma_con_num, soma_con_ind, path_node, del_neurite_ma )
%This function is used to search all the child neurite in the common path
%and simplify the segmentation problem.

m = soma_con_num;
n2 = length(path_node);
fitness_matrix = zeros(n2,m);
fitness_matrix(:, 1) = path_node;
connect_ind = cell(1, m);
soma_con = soma_con_ind;


for i = 1:1:m
    del_ind = zeros(n2, 1);
    connect_ind_1 = connect_ma{soma_con(i)};    
    connect_ind{i} = zeros(n2, 2);
    for j = 1:1:n2
        aa = find_vector_beta_1(Neuron{i},Neurite_matrix(path_node(j),:));
        if length(aa) > 1
            error('Two same branches!');
        end
        del_ind(j) = find_vector_beta_1(Neuron{i},Neurite_matrix(path_node(j),:));
        connect_ind{i}(j, :) =  connect_ind_1( connect_ind_1(:, 1) == path_node(j), 1:2);
    end
    connect_ind_2 = connect_ind{i};
    connect_ind_2(~ismember(connect_ind_2(:,2), path_node), :) = [];
    connect_ind{i} = connect_ind_2;
    fitness_matrix(:, i) = total_fitness_ma(path_node, soma_con(i));
   del_neurite_ma{soma_con(i)} = [del_neurite_ma{soma_con(i)}; del_ind];
end

end


