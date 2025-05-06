function  [node_weight ] = neu_length(neurite_matrix, location_matrix, radias_matrix)
%This function is used for computing the neurite length as the weight of
%each neurite.
[m n] = size(neurite_matrix);
node_weight = zeros(m,2); 

for i = 1:1:m
    radias = 0;
    eu = 0;
    node = neurite_matrix(i,:);
    node(isnan(node)) = [];
    for j = 1:1:length(node)-1
        eu = eu + sqrt((location_matrix(node(j),2)-location_matrix(node(j+1),2))^2 + (location_matrix(node(j),3)-location_matrix(node(j+1),3))^2 + (location_matrix(node(j),4)-location_matrix(node(j+1),4))^2);
    end
    for j = 1:1:length(node)
        radias = radias + radias_matrix(node(j));
    end
    av_radias = radias/length(node);
    node_weight(i,1) = eu;
    node_weight(i,2) = av_radias;
end