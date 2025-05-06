function [soma_node path_node] = find_connection(path_node_set, soma_node_set, connect_matrix)
%This function is used for finding the connection point between path set
%and soma set
soma_node = [];
path_node = [];
n = length(soma_node_set);
for i =1:1:n
    child_set = find(connect_matrix(soma_node_set(i),:)==1);
    child_con = ismember(path_node_set,child_set);
    child_node  = path_node_set(child_con);
    if ~isempty(child_node)
        soma_node = [soma_node; soma_node_set(i)];
        path_node = [path_node; child_node];
    end
end
    