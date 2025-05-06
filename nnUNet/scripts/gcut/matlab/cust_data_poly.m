function [ poly_parameter ] = cust_data_poly( poly_path )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    whole_GOF_ma = [];
    
    filelist = dir(fullfile(poly_path));
    
    filelist(ismember({filelist.name},{'.','..'})) = [];
    
    if isempty(filelist)
        
        error('No customized data!');
                
    end
    
    count = zeros(length(filelist), 1);
    
    %fs={'\b\b%d%%','\b\b\b%d%%'};
    hwait = waitbar(0, 'wait>>>>>>>>>>');
    %fprintf('0%%');
    
    for fi_i = 1:1:length(filelist)
        
        filename = filelist(fi_i).name;
        
        load_filename = strcat(poly_path, filename);
        
        GOF_matrix = GOF_feature_beta(load_filename);
        
        whole_GOF_ma = [whole_GOF_ma, GOF_matrix']; %检测一致性
        
        waitbar(fi_i/length(filelist), hwait, strcat('Processing ', ...
            num2str(fix(fi_i/length(filelist)*100)), '%') );%显示进度
            
        %p_i = fix(fi_i/length(filelist) *100);
       
            
        %fprintf(fs{1+(p_i>9)},p_i)
        
    end
    
    %fprintf('\n');
    close(hwait);
    whole_GOF_ma(:, whole_GOF_ma(3,:) == 1) = [];
    
    bin = floor(length(filelist)/10);
    
    if bin > 200
        
        bin = 200;
        
    elseif bin<10
        
        bin = 10;
    
    end
    
    whole_GOF = whole_GOF_ma(2, :);
    
    GOF_dist = hist(whole_GOF, bin);
    
    GOF_dist = GOF_dist/length(whole_GOF);
    
    x_1 = 0:pi/(bin-1):pi;
    
    if bin>20
        
        [p,S,mu] = polyfit(x_1,GOF_dist,20);
        
    else
        
        [p,S,mu] = polyfit(x_1,GOF_dist,bin - 1);
    end
    
    GOF_polyfit.p = p;
    
    GOF_polyfit.S = S;
    
    GOF_polyfit.mu = mu;
    
    poly_parameter = GOF_polyfit;
    
end

