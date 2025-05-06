function [data ] = swc_read(filename)
%This function is used to read the data from swc file.
fid = fopen(filename, 'r');
headerline = 0;
while ~feof(fid)
    tline=fgetl(fid);
    if ~isempty(tline)
        if double(tline(1))==35  
            headerline = headerline + 1;
        end
    else
        headerline = headerline + 1;
    end
end
fclose(fid);
DELIMITER = ' ';
Structure_data = importdata(filename, DELIMITER, headerline);
if isstruct(Structure_data)==1
    data = Structure_data.data;
else
    data = Structure_data;
end