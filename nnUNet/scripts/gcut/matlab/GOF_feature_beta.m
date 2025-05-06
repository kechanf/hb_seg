function [ neurite_an_vector ] = GOF_feature_beta( filename )

%path: Load file path
%x_convert:the x scale to convert pixel to micrometer
%y_convert:the y scale to convert pixel to micrometer
%z_convert:the z scale to convert pixel to micrometer

    S_data = swc_read(filename);
    
    A = S_data;
    
    [m,n] = size(A);
    
    B = A(:,3:5);
    
    vector_matrix = zeros(m,9); %矢量矩阵，记录结点上每个点的书序（1），空间坐标（2:4），该点与父节点的方向矢量（5:7），该点的父节点（8），以及该点矢量与标准矢量的夹角（9）。
    
    vector_matrix(:,1) = A(:,1);
    
    vector_matrix(:,2:4) = B;
    
    vector_matrix(:,8) = A(:,7);
    
    standard_vector = zeros(m,3); %标准矢量，记录从soma到该点的矢量

    for i =m:-1:1
        
        if (A(i,7) == -1)
            
            continue;
            
        else
            index_1 = A(i,7);
            
            vector_matrix(i,5) = (B(i,1)-B(index_1,1));%该点的矢量为父节点到该点的方向矢量，即该点位置与父节点的差值
            
            vector_matrix(i,6) = (B(i,2)-B(index_1,2));
            
            vector_matrix(i,7) = (B(i,3)-B(index_1,3));
            
            standard_vector(i,1) = (B(index_1,1)-B(1,1)); %该点的标准矢量由该点的父节点与soma的位置来决定
           
            standard_vector(i,2) = (B(index_1,2)-B(1,2));
            
            standard_vector(i,3) = (B(index_1,3)-B(1,3));
            
            vector_matrix(i,9) = acos(((vector_matrix(i,5)*standard_vector(i,1) + vector_matrix(i,6)*standard_vector(i,2) + vector_matrix(i,7)*standard_vector(i,3))/...
                (sqrt(vector_matrix(i,5)^2 + vector_matrix(i,6)^2 + vector_matrix(i,7)^2)*sqrt(standard_vector(i,1)^2 + standard_vector(i,2)^2 + standard_vector(i,3)^2))));
            
            vector_matrix(isnan(vector_matrix))=0;
        
        end
        
    end
    %-----------------------------------------------------------------
    Parent_list = ones(m,2);
    
    Parent_list(:,1) = A(:,1);
    
    Parent_list(:,2) = A(:,7);
    
    Child_list = ones(m,2);
    
    Child_list(:,1) = A(:,7);
    
    Child_list(:,2) = A(:,1);
    
    Child_list((Child_list(:,1) == -1),:) = [];
    
    elem_A = A(:,1);
    
    [count_elem, node_ind] = hist(Child_list(:,1),elem_A);
    
    leaf_node = node_ind(count_elem == 0);
    
    branch_node = node_ind(count_elem > 1);

    
    %------------------------------------------------------------------

    if ~isempty(branch_node)
        
        [raw_i raw_j] = size(A);
        
        location_ma = zeros(raw_i,4);
        
        location_ma(:,1) = A(:,1);
        
        location_ma(:,2:4) = A(:,3:5);
        
        diam_ma = A(:,6);
        
        [neurite_ma] = split_neurite_delta(branch_node,leaf_node,Parent_list, Child_list);
        
        [era_i era_j] = size(neurite_ma);
        
        for pre_i =1:1:era_i
            
            era = find(neurite_ma(pre_i,:)==0);
            
            neurite_ma(pre_i,era) = NaN;
        
        end
        
        [neurite_parem ]  = neu_length(neurite_ma, location_ma,diam_ma);
        
        [neurite_an_vector,neurite_child]= feature_detect_beta(vector_matrix,location_ma,Child_list,Parent_list,branch_node,leaf_node,neurite_ma, 1); %进行剪枝
        
        for pre_i =1:1:era_i
           
            era = isnan(neurite_child(pre_i,:));
           
            neurite_child(pre_i,era) = 0;
        
        end
        
        neurite_an_vector = [neurite_an_vector, neurite_parem];
        
    else
        
        neurite_an_vector = [];
        
        neurite_child = [];
    end


end

