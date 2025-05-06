function [ matrix ] = tree_resort( raw_matrix )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    for ii_1 = 2:1:length(raw_matrix(:, 1))
        ind = find(raw_matrix(:, 1) == raw_matrix(ii_1, 7));
        raw_matrix(ii_1, 7) = ind(1);
    end
    raw_matrix(:, 1) = 1:length(raw_matrix(:, 1));
    raw_matrix(1, 7) = -1;
    matrix = raw_matrix;

end

