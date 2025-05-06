function [Neuron neurite_index] = path_extract_beta_1(Neu_con_ma, Neuron, neurite_matrix, soma_set, soma_connect, orin_soma_node, neurite_index, previous_ma, fitness_ma)
%This function is used to find the common path between somas and use the
%topological constraints to find the neurite assignment for each soma.
[neu_in_r neu_in_c] = size(neurite_index);
path_neu_ind = neurite_index;
for neu_in_i = 1:1:neu_in_c
    path_neu_ind(:, neu_in_i) = path_neu_ind(:, neu_in_i).* (2^(neu_in_i - 1));
end
to_path_ind = sum(path_neu_ind, 2);
to_path_ind = [to_path_ind, zeros(neu_in_r, 1)];
cell_ind = 1;
while(~isempty(find(to_path_ind(:, 2) == 0)))
    unsig_node = find(to_path_ind(:, 2) == 0);
    unsig_node_1 = unsig_node(1);
    com_node_set_1 = find(to_path_ind(:, 1) == to_path_ind(unsig_node_1, 1));
    if length(find(neurite_index(unsig_node_1, :) == 1)) > 1   
        in_com_path_node_cell{cell_ind} = com_node_set_1;
        path_soma_set_cell{cell_ind} = neurite_index(unsig_node_1, :);
        neurite_index(com_node_set_1, :) = 0; %------------------------------
        cell_ind = cell_ind + 1;
    end
    to_path_ind(com_node_set_1, 2) = 1;
end

[m n] = size(soma_set);
soma_connect = soma_connect - eye(m);
logic_Neu_con_ma = Neu_con_ma;
[m2 n2] = size(Neu_con_ma);
for i = 1:1:n2
    logic_Neu_con_ma{i}(logic_Neu_con_ma{i}>0) = 1; 
end
[p_m p_n] = size(in_com_path_node_cell);
path_assin_struct_cell = cell(p_n,1);
disp('common path end');
Neuron_de_ma = cell(1, m);
for i = 1:1:p_n
    in_path_soma_con = path_soma_set_cell{i};
    in_path_node = in_com_path_node_cell{i};
    [h v] = size(in_path_node);
    if h<v
        in_path_node = in_path_node';
    end
    soma_con_ind = find(in_path_soma_con==1);
    soma_con_num = length(soma_con_ind);
    neuron_con_ma_cell = cell(1,soma_con_num);
    logic_con_ma_cell = cell(1,soma_con_num);
    in_soma_orin_cell = cell(1,soma_con_num);
    Neuron_in = cell(1,soma_con_num);
    for ce_i = 1:1:soma_con_num %store the information of common path
        neuron_con_ma_cell{ce_i} = Neu_con_ma{soma_con_ind(ce_i)};
        in_soma_orin_cell{ce_i} =  orin_soma_node{soma_con_ind(ce_i)};
        Neuron_in{ce_i} =  Neuron{soma_con_ind(ce_i)};
        logic_con_ma_cell{ce_i} = logic_Neu_con_ma{soma_con_ind(ce_i)};
    end
    [in_fitness_ma, in_connect_ind, Neuron_de_ma] = neurite_route_sort_gamma_2(neurite_matrix, previous_ma, Neuron_in, fitness_ma, soma_con_num, soma_con_ind, in_path_node, Neuron_de_ma );
    %defint the structure to store the processed common path information
    path_struct.fitness = in_fitness_ma;
    path_struct.connect_ind = in_connect_ind;
    path_struct.path_node = in_path_node;
    path_struct.connect_soma = soma_con_ind;
    path_struct.connect_soma_num = length(soma_con_ind);
%    path_struct.path_child_cell = in_path_child_cell;
    path_assin_struct_cell{i} = path_struct;
    disp(num2str(i));
end

for ce_i = 1:1:m
    Neuron{ce_i}(Neuron_de_ma{ce_i},:) = [];
end
    
for i = 1:1:p_n %Use linear programming with topological constraints to determine the neurite assignment
    [assign_ma] = linear_path_assign_pre(path_assin_struct_cell{i});
    m = path_assin_struct_cell{i}.connect_soma_num;
    neuron_id = path_assin_struct_cell{i}.connect_soma;
    path_node = path_assin_struct_cell{i}.path_node;
    [max_Y max_I] = max(assign_ma,[],2);
    for ii = 1:1:m 
        assign_node = (max_I==ii);
        recon_node = path_node(assign_node);
        neurite_index(recon_node, neuron_id(ii)) = 1; 
        [new_Neuron ] = path_reconnect_beta(recon_node, Neuron{neuron_id(ii)}, neurite_matrix);
        Neuron{neuron_id(ii)} = new_Neuron; 
    end
end