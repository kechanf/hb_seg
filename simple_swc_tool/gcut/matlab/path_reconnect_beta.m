function [new_Neuron ] = path_reconnect_beta( path_node_soma, Neuron, neurite_matrix)
%This function is used to reconnect the Child node of path node to the
%path.
soma_path_neu = path_node_soma;
[m_5] = length(path_node_soma);

new_Neuron = Neuron;

for i_10 = 1:1:length(soma_path_neu)
neurite_index = soma_path_neu(i_10);
add_neu = neurite_matrix(neurite_index,:);
new_Neuron = [new_Neuron;add_neu];
end