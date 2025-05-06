function [] = save_file_std_swc(Save_name, matrix)
%This function is used for saving the standard swc file 
  fid=fopen(Save_name,'wt');
            [m,n]=size(matrix);  
            for save_i=1:1:m     
                for save_j=1:1:n        
                    if save_j==n           
                        fprintf(fid,'%d\n',matrix(save_i,save_j));       
                    else
                        if save_j==1 || save_j==2 || save_j==7
                            fprintf(fid,'%d ',matrix(save_i,save_j));
                        else
                            fprintf(fid,'%g ',matrix(save_i,save_j));
                        end
                    end
                end
            end
            fclose(fid);
