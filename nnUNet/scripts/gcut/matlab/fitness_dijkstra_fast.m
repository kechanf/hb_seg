function [ fitness_matrix, in_neuron, connect_ma, soma_connect, in_neurite_ind, previous_matrix ] = fitness_dijkstra_fast( neurite_matrix, location_matrix, branch_ma, in_neuron, graph_soma_node, soma_neurite, soma_set, soma_ind, neurite_weight, poly_para )
%   This function used dijkstra algorithm to search all branches in the
%   connecting graph, break the graph into directed acyclic graph, 
%   and caculate fitness for each branch-soma pair.
%   Detailed explanation goes here
soma_connect = zeros(length(soma_set), length(soma_set));
n = length(neurite_matrix(:, 1));
fitness_matrix = ones(n, 1).* -inf;
connect_ma = sparse(zeros(n, n));
d = inf*ones(1,n); % distance s-all nodes
d(graph_soma_node) = 0;    % s-s distance
T = 1:n;    % node set with shortest paths not found
soma_1 = union(neurite_matrix(graph_soma_node,:), neurite_matrix(graph_soma_node,:));
soma_1(isnan(soma_1)) = [];
soma = soma_set(soma_ind); 
pre_ma = zeros(n, 4);  %1. number of node,   2.previous node,   3.distance to previous node.  4.connect node
pre_ma(:,1) = (1:n)';
node_index = zeros(length(location_matrix(:, 1)), 1); %node index to ensure the topological connection
node_index(soma_1) = 1;
mask_ma = 1:n;
flag = 1;
while not(isempty(flag))
    [dmin,ind] = min(d(T));
    if length(T) == 367
        aaa = 1;
    end
    neurite_ind = neurite_matrix(T(ind),:);
    neurite_ind(isnan(neurite_ind)) = [];
    T_neurite = neurite_matrix;
    T_neurite(setdiff(mask_ma, T),:) = nan;
    j = neurite_connection(neurite_ind, T(ind), T_neurite); 
    if isempty(j) 
        T = setdiff(T,T(ind));
    else
        for ii = 1:1:length(j(:,1))
            con_node = j(ii, 2);  
            
            con_neu_ind = j(ii, 1);
            if soma_neurite(con_neu_ind) ~=0 && soma_neurite(con_neu_ind) ~= soma_ind
                  fitness_matrix(con_neu_ind) = inf;
                  soma_connect(soma_ind, soma_neurite(con_neu_ind)) = 1;
           else
            neurite_1 = neurite_matrix(con_neu_ind,:);
            neurite_1(isnan(neurite_1)) = [];
            m_1 = length(neurite_1);
                if length(con_node) == 1
                    ind_1 = find(neurite_1 == con_node);
                    re_node_1 = rest_node(neurite_1, ind_1);
                    ind_2 = find(neurite_ind == con_node);
                    re_node_2 = rest_node(neurite_ind, ind_2);
                    con_ind = node_index(re_node_1) | node_index(re_node_2);
                    if con_ind ~= 0
                        branch_order = branch_ma(T(ind)) + 1; 
                        total_an = 0;
                        if ind_1 == 1
                            for ind_j = m_1 - 1: -1: 1
                                neu_ve = location_matrix(neurite_1(ind_j + 1), 2:4) - location_matrix(neurite_1(ind_j), 2:4);
                                std_ve = location_matrix(neurite_1(ind_j), 2:4) - location_matrix(soma, 2:4);
                                com_an = acos(sum(neu_ve .* std_ve)/(sqrt(sum(neu_ve.^2))*sqrt(sum(std_ve.^2))));
                                if ~isreal(com_an)
                                    com_an = real(com_an);
                                end
                                if isnan(com_an)
                                    com_an = pi;
                                end
                                total_an = total_an + com_an;
                            end
                        elseif ind_1 == m_1
                            for ind_j = 1:1:m_1-1
                                neu_ve = location_matrix(neurite_1(ind_j), 2:4) - location_matrix(neurite_1(ind_j +1), 2:4);
                                std_ve = location_matrix(neurite_1(ind_j + 1), 2:4) - location_matrix(soma, 2:4);
                                com_an = acos(sum(neu_ve .* std_ve)/(sqrt(sum(neu_ve.^2))*sqrt(sum(std_ve.^2))));
                                if ~isreal(com_an)
                                    com_an = real(com_an);
                                end
                                if isnan(com_an)
                                    com_an = pi;
                                end
                                total_an = total_an + com_an;
                            end
                        end
                        av_an = total_an/(m_1 - 1);
                        fitness_matrix(con_neu_ind) = Gassi_com_fit_beta(av_an, branch_order, poly_para);
                        if fitness_matrix(con_neu_ind) == inf
                              disp('error fitness inf!');
                              disp(strcat(num2str(av_an),',',num2str(branch_order)));
                        end
                        adj_fit = 1/(fitness_matrix(con_neu_ind)); %fitness based penaty
                        if d(con_neu_ind)>d(T(ind))+adj_fit
                            d(con_neu_ind)=d(T(ind))+adj_fit;
                            if pre_ma(con_neu_ind, 2) ==0
                                pre_ma(con_neu_ind, 2) = T(ind);
                                pre_ma(con_neu_ind, 3) = (1/adj_fit)*neurite_weight(con_neu_ind,2);%*2*(1+(neurite_weight(T(ind),2)-neurite_weight(con_neu_ind,2))...
                               %/(neurite_weight(T(ind),2)+neurite_weight(con_neu_ind,2)));
                               % use branch length as weight
                                pre_ma(con_neu_ind, 4) = con_node;
                                node_index(con_node) = 1;
                                branch_ma(con_neu_ind) = branch_ma(T(ind)) + 1;
                                in_neuron = [in_neuron; neurite_matrix(con_neu_ind, :)];
                            else
                                pre_ma(con_neu_ind, 2) = T(ind);
                                pre_ma(con_neu_ind, 3) = (1/adj_fit)*neurite_weight(con_neu_ind,2);%*2*(1+(neurite_weight(T(ind),2)-neurite_weight(con_neu_ind,2))...
                               %/(neurite_weight(T(ind),2)+neurite_weight(con_neu_ind,2)));
                               % use branch length as weight
                                pre_ma(con_neu_ind, 4) = con_node;
                                node_index(con_node) = 1;
                                branch_ma(con_neu_ind) = branch_ma(T(ind)) + 1;
                            end
                        end
                    end
                end
            end
        end
        T = setdiff(T,T(ind));
    end 
    flag = find(d(T) ~=inf);
end

in_neurite_ind = zeros(n, 1);

in_neurite_ind(graph_soma_node) = 1;

for c_i = 1:1:n
    if pre_ma(c_i, 2) ~=0
        connect_ma(pre_ma(c_i, 2), pre_ma(c_i, 1)) = pre_ma(c_i, 3);
        in_neurite_ind(pre_ma(c_i, 1)) = 1;
        in_neurite_ind(pre_ma(c_i, 2)) = 1;
    end
end

previous_matrix = pre_ma;





end

