function [node_1 ] = rest_node(vector, node_ind)

n = length(vector);

if node_ind == 1
    
    node_1 = vector(n);
    
else
    
    node_1 = vector(1);
    
end
