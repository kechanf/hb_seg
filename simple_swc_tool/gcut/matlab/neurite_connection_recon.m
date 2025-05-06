function [ con_neu_ind ] = neurite_connection_recon( neurite, ind_1, neurite_matrix, location_ma )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
dist_c = 10;

con_neu_ind = [];

node_1 = neurite(1);

node_2 = neurite(end);

node_set = union(neurite_matrix, neurite_matrix);

node_set(isnan(node_set)) = [];

dist_ma_1 = sqrt(sum((repmat(location_ma(node_1, 1:3), length(node_set(:, 1)), 1) - location_ma(node_set(:, 1), 1:3)).^2, 2));


c_node_1 = node_set(dist_ma_1 < dist_c);


dist_ma_2 = sqrt(sum((repmat(location_ma(node_2, 1:3), length(node_set(:, 1)), 1) - location_ma(node_set(:, 1), 1:3)).^2, 2));

c_node_2 = node_set(dist_ma_2 < dist_c);

m_row = [];

for c_i = 1:1:length(c_node_1)
    
    [m_row_1 m_col] = find(neurite_matrix == c_node_1(c_i));
    
    m_row = [m_row; m_row_1];
    
end

n_row = [];

for c_i = 1:1:length(c_node_2)

    [n_row_1 n_col] = find(neurite_matrix == c_node_2(c_i));
    
    n_row = [n_row; n_row_1];
    
end

m_row(m_row == ind_1) = [];

n_row(n_row == ind_1) = [];

con_ind_1 = [m_row, ones(length(m_row), 1).*node_1];

con_ind_2 = [n_row, ones(length(n_row), 1).*node_2];

con_neu_ind = [con_ind_1; con_ind_2];


end

