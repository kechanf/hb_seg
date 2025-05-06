function [ recon_Neuron ] = neurite_toporecon_ver_1( Neuron,  connect_cell, previous_cell, orin_node, soma_set, raw_matrix, neurite_matrix, neurite_index_ma)
%This function is used to reconstruct branches and a given soma into a
%neuron
%   Detailed explanation goes here
recon_Neuron = cell(size(Neuron));
for i = 1:1:length(Neuron)
    neurite_ma = Neuron{i};
    if isempty(neurite_ma)
        continue;
    else
        connect_ma = connect_cell{i};
        previous_ma = previous_cell{i};
        soma = soma_set(i);
        soma_node = orin_node{i};
        recon_ma = raw_matrix(soma, :); 
        recon_ma(1, 7) = -1;
        neurite_ind = find(neurite_index_ma(:, i) == 1);
        connect_ma_1 = zeros(size(connect_ma));
        connect_ma_1(neurite_ind, neurite_ind) = connect_ma(neurite_ind, neurite_ind);

        for i_1 = 1:1:length(soma_node)
            neu_1 = neurite_matrix(soma_node(i_1),:);
            neu_1(isnan(neu_1)) = [];
            ind_1 = find(neu_1 == soma);
            if ind_1 == length(neu_1)
                ma = [neu_1(end -1 :-1: 1)', raw_matrix(neu_1(end -1:-1:1), 2:6), neu_1(end: -1: 2)'];
                recon_ma = [recon_ma; ma];
            elseif ind_1 == 1
                ma = [neu_1(2:end)', raw_matrix(neu_1(2 :end), 2:6), neu_1(1:end -1)'];
                recon_ma = [recon_ma; ma];
            else
                ma = [neu_1(ind_1 -1 :-1: 1)', raw_matrix(neu_1(ind_1 -1:-1:1), 2:6), neu_1(ind_1: -1: 2)'];
                ma_1 = [neu_1(ind_1 + 1:end)', raw_matrix(neu_1(ind_1 + 1 :end), 2:6), neu_1(ind_1:end -1)'];
                recon_ma = [recon_ma; ma; ma_1];
            end
        end

        [or_node, neigh_1] = find(connect_ma_1(soma_node, :) ~=0);
        or_node = soma_node(or_node);
        neigh_1 = intersect(neigh_1, neurite_ind);
        while(~isempty(neigh_1))
            for j = 1:1:length(neigh_1)
                neu_1 = neurite_matrix(neigh_1(j),:);
                neu_1(isnan(neu_1)) = [];
                con_node = previous_ma(intersect(find(previous_ma(:, 1) == neigh_1(j)), find(previous_ma(:, 2) == or_node(j))), 4);
                if isempty(con_node) || length(con_node) > 1
                    dist = pdist2(raw_matrix(neu_1, 3:5), raw_matrix(recon_ma(:, 1), 3:5));
                    min_dist = min(min(dist));
                    [ind_1, ind_2] = find(dist == min_dist);
                    ma = [neu_1(ind_1), raw_matrix(ind_1, 2:6), recon_ma(ind_2, 1)];
                    recon_ma = [recon_ma; ma];
                else
                ind_1 = find(neu_1 == con_node);
                end
                if ind_1 == length(neu_1)
                   ma = [neu_1(end -1 :-1: 1)', raw_matrix(neu_1(end -1:-1:1), 2:6), neu_1(end: -1: 2)'];
                   recon_ma = [recon_ma; ma];
                elseif ind_1 == 1
                   ma = [neu_1(2:end)', raw_matrix(neu_1(2 :end), 2:6), neu_1(1:end -1)'];
                   recon_ma = [recon_ma; ma];
                end
            end
            [or_node, neigh_2] = find(connect_ma_1(neigh_1, :) ~=0);
            or_node = neigh_1(or_node);
            neigh_1 = neigh_2;
            neigh_1 = intersect(neigh_1, neurite_ind);
        end
        [recon_ma ] = tree_resort(recon_ma);
        recon_Neuron{i} = recon_ma;
    end
end
    
end




