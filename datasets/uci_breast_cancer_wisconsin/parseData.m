% Parses the breast cancer data, normalizes it and stores it in matlab
% format
%
% @author Wolfgang Roth
% @date created on 30.10.2015

close all;
clear all;

fid = fopen('wdbc.data');
tline = fgets(fid);
data = [];
while ischar(tline) && length(tline) > 5
    pline = sscanf(tline, '%f,%c,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f');
    pline = pline';
    if pline(2) == 'M'
      pline = [pline(3:end), 0];
    elseif pline(2) == 'B'
      pline = [pline(3:end), 1];
    else
      fprintf('"%s"\n', tline);
      error('Could not parse class: %c', pline(2));
    end
    data = [data; pline];
    tline = fgets(fid);
end
fclose(fid);

clear fid tline pline tmp;
x = data(:, 1:end-1);
t = data(:, end);
x = bsxfun(@times, bsxfun(@minus, x, mean(x)), 1 ./ std(x));

save('breast_cancer_wisconsin_diagnostic.mat', 'x', 't');
