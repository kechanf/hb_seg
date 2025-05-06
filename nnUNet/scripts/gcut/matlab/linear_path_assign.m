function [assign_ma ] = linear_path_assign(fitness_matrix, connect_ind, path_node, connect_soma_num)
%This function uses linear programming method to find the global minimum of
%total penalty.
m = connect_soma_num;
weight_cell = cell(1,m);
n = length(path_node);
[f_m f_n ] = size(fitness_matrix);
assign_ma = zeros(f_m,f_n);
for i = 1:1:m
    in_connect_ind = connect_ind{i};
    [m_1 n_1] = size(in_connect_ind);
    a_1 = 1:1:m_1;
    a_1 = a_1';
    a_1 = [a_1;a_1];
    con_next = zeros(m_1,1);
    con_pre = zeros(m_1,1);
    for j = 1:1:m_1
        con_next(j) = find(path_node == in_connect_ind(j,1));
    end
    for j = 1:1:m_1
        con_pre(j) = find(path_node == in_connect_ind(j,2));
    end
    con_ma = [con_next;con_pre];
    ind_next = ones(m_1,1).*1;
    ind_pre = ones(m_1,1).*-1;
    ind_ma = [ind_next;ind_pre];
    weight_ma = sparse(a_1,con_ma,ind_ma,m_1,n);
    weight_cell{i} = weight_ma;
end

for i_1 = 1:1:m
    if i_1 == 1
        A = weight_cell{1};
    else
        [m_2 n_2] = size(weight_cell{i_1});
        [m_3 n_3] = size(A);
        A = [A, sparse(m_3,n_2); sparse(m_2,n_3),weight_cell{i_1}];
    end
end

[m_3 n_3] = size(A);
b = zeros(m_3,1);

I = eye(n,n);
Aeq = [];
for i_1 = 1:1:m
    Aeq = [Aeq,I];
end

beq = ones(n,1);

lb = zeros(n*m,1);
ub = inf(n*m,1);
f = [];
for f_i = 1:1:f_n
    fit_ma = fitness_matrix(:,f_i);
    f = [f;fit_ma];
end

[x fval] = linprog(f,A,b,Aeq,beq,lb,ub);

for st_i = 1:1:m
    assign_ma(:,st_i) = x((st_i-1)*f_m+1:st_i*f_m);
end





