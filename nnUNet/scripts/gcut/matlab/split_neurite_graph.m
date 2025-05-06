function [ neurite_matrix ] = split_neurite_graph( cluster_graph )
%[ neurite_matrix ] = split_neurite_graph( cluster_graph )
% This function is used to split the neuron cluster denoted as a conneted 
%graph into branches by using the graph theory. 
% Input: cluster_graph - Neuron clusters denoted as a connected graph. The
% nodes in the graph represent the digital nodes in the reconstructed
% neuron cluster and the edges in the graph represent the connections
% between digital nodes.
%
%Output: neurite_matrix: Branch matrix, which represents all branches
%splited from the neuron cluster
    
    graph_1 = cluster_graph;
    
    graph_1 = symmetrize(graph_1);
    
    neigh_n = sum(graph_1, 2);
    
    branch_node = find(neigh_n > 2);
    
    con_node = find(neigh_n == 2);
    
    leaf_node = find(neigh_n == 1);
    
    ind_ma = [ (1:length(neigh_n))', neigh_n];
    
    i_c = 1;
    
    branch_ma = branch_node;
    
    while(~isempty(branch_ma))
        
        s_node = branch_ma(1);
        
        neurite_1 = s_node;
        
        neigh_1 = find(graph_1(s_node, :) == 1);
        
        neigh_1(neigh_1 == s_node) = [];
        
        neigh_1 = neigh_1(1);
        
        graph_1(s_node, neigh_1) = 0;
        
        graph_1(neigh_1, s_node) = 0;
        
        neurite_1 = [neurite_1, neigh_1];
        
        while (isempty(find(branch_node == neigh_1, 1)) && isempty(find(leaf_node == neigh_1, 1)));
            
            neigh_2 = find(graph_1(neigh_1, :) == 1);
            
            neigh_2(neigh_2 == s_node) = [];
            
            s_node = neigh_1;
            
            neigh_1 = neigh_2(1);
            
            graph_1(s_node, neigh_1) = 0;
        
            graph_1(neigh_1, s_node) = 0;
            
            neurite_1 = [neurite_1, neigh_1];
            
        end
        
        del_branch = find(sum(graph_1(branch_ma, :), 2) == 0);
        
        branch_ma(del_branch) = [];
        
        for j_c = 1:1:length(neurite_1)
            
            neurite_matrix(i_c, j_c) = neurite_1(j_c);
            
        end
        
        i_c = i_c + 1;
        
    end
        
        


end

