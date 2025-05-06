function [Neuron, re_Neuron ] = connect_neurite_beta_3(raw_matrix, neurite_matrix, Set_index, poly_para)

[raw_i raw_j] = size(raw_matrix);

% [node_id, x, y, z]
location_matrix = zeros(raw_i,4);

location_matrix(:,1) = raw_matrix(:,1);

location_matrix(:,2:4) = raw_matrix(:,3:5);

% radius
radia_matrix = raw_matrix(:,6);

[m, n] = size(neurite_matrix); 

soma_set = find(Set_index==1); 

[m1, n1] = size(soma_set);

Neuron = cell(n1,m1);

orin_soma_node = cell(n1,m1);

Neu_con_ma = cell(n1,m1); 

previous_ma = cell(n1, m1);

fitness_ma = zeros(m,m1); 

sort_neurite_ma = zeros(m,1); 

[node_weight ] = neu_length(neurite_matrix, location_matrix, radia_matrix); 

branch_ma = zeros(m, 1);

for soma_i = 1:1:m1    %initialize the graph
    
    for neu_i = 1:1:m
        
        ne_ind = (find(neurite_matrix(neu_i,:)==soma_set(soma_i)));
        
            if ~isempty(ne_ind)
                
                sort_neurite_ma(neu_i) = soma_i;
                
                set_neu_ma = neurite_matrix(neu_i,:);
                
                set_neu_ma(isnan(set_neu_ma))=[];
                
                Set_index(set_neu_ma) = 1;
                
                Neuron{soma_i} = [Neuron{soma_i};neurite_matrix(neu_i,:)]; 
                
                branch_ma(neu_i) = 1;
                
            end
    end
    
end

soma_connect = zeros(m1,m1);

for or_i = 1:1:m1
    
    for or_j = 1:1:length(Neuron{or_i}(:,1))
        
        orin_so_1 = find_vector_beta_1(neurite_matrix,Neuron{or_i}(or_j,:));
        
        if isempty(orin_so_1) || length(orin_so_1) > 1
            error('Original branch is not unique!');
        end
        
        orin_soma_node{or_i}(or_j) = find_vector_beta_1(neurite_matrix,Neuron{or_i}(or_j,:));
        
    end
    
end

neurite_index = zeros(m, m1); 

for so_i = 1:1:m1
    sub_neuron = Neuron{so_i};
    sub_sort_nerite_ma = sort_neurite_ma;
    [sub_fitness_ma, sub_neuron, sub_con_ma, sub_soma_connect, in_neurite_ind, pre_ma] = fitness_dijkstra_fast(neurite_matrix, location_matrix, branch_ma, sub_neuron, orin_soma_node{so_i}, sub_sort_nerite_ma, soma_set, so_i, node_weight, poly_para);
    Neu_con_ma{so_i} = sub_con_ma;
    Neuron{so_i} = sub_neuron;
    fitness_ma(:, so_i) = sub_fitness_ma;
    previous_ma{so_i} = pre_ma;
    soma_connect = soma_connect | sub_soma_connect;
    disp(num2str(so_i));
    neurite_index(:, so_i) = in_neurite_ind;
end
    
disp('connection graph construction complete!');

[Neuron, neurite_index] = path_extract_beta_1(Neu_con_ma, Neuron, neurite_matrix, soma_set, soma_connect, orin_soma_node, neurite_index, previous_ma, fitness_ma); 

disp('path assign complete!');

[re_Neuron] = neurite_toporecon_ver_1(Neuron,  Neu_con_ma, previous_ma, orin_soma_node, soma_set, raw_matrix, neurite_matrix, neurite_index);