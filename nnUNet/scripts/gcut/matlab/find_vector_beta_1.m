function [index ] = find_vector_beta_1(Matrix, vector)
%This function is used to find the location of vector in the matrix
index = [];
vector(isnan(vector)) = [];
n = length(vector);
id_ma = cell(1, n);
for i = 1:1:n
    id_ma{i} = find(Matrix(:,i)==vector(i));
end
id = id_ma{1};
for i = 2:1:n
    id = intersect(id, id_ma{i});
end
if ~isempty(id)
    index = [index; id];
end