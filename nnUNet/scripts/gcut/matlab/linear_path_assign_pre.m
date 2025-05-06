function [assign_ma] = linear_path_assign_pre(path_struct)
%This function is used to preprocess the data for linear programming.
if length(path_struct.path_node) == 1
    assign_ma = zeros(size(path_struct.fitness));
    [~, ind] = max(path_struct.fitness);
    assign_ma(ind) = 1;
else
    fitness_matrix = -path_struct.fitness;
    connect_ind = path_struct.connect_ind;
    path_node_matrix = path_struct.path_node;
    connect_soma_num = path_struct.connect_soma_num;

    [ assign_ma ] = linear_path_assign( fitness_matrix, connect_ind, path_node_matrix, connect_soma_num );
end