function [fitness] = Gassi_com_fit_beta(angle, order, poly_para)
%This function is used to compute fitness of a neurite for a given soma
%according to the branch order and angle between neurite and soma.

p = poly_para.p;

mu = poly_para.mu;

fun = @(x)polyval(p, x, [], mu);

cdf_sum = integral(fun, 0, pi);

a = 1/cdf_sum;

fitness = a* integral(fun, angle, pi);
fitness = fitness * (1+log(1+1/order))*4;


end


