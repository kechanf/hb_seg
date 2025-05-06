function [ con_neu_ind ] = neurite_connection( neurite, ind_1, neurite_matrix )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
con_neu_ind = [];

node_1 = neurite(1);

node_2 = neurite(end);

[m_row m_col] = find(neurite_matrix == node_1);

[n_row n_col] = find(neurite_matrix == node_2);

m_row(m_row == ind_1) = [];

n_row(n_row == ind_1) = [];

con_ind_1 = [m_row, ones(length(m_row), 1).*node_1];

con_ind_2 = [n_row, ones(length(n_row), 1).*node_2];

con_neu_ind = [con_ind_1; con_ind_2];

% for i = 1:1:length(neurite_matrix(:,1))
%     
%     neurite_1 = neurite_matrix(i,:);
%     
%     con_node = intersect(neurite_1, neurite);
%     
%     if ~isempty(con_node) && length(con_node) == 1
%         
%         ind = [i, con_node];
%         
%         con_neu_ind = [con_neu_ind ; ind];
%         
%     end
%     
% end

end

