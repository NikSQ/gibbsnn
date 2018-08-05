% Parses the iris data, normalizes it and stores it in matlab format
%
% @author Wolfgang Roth
% @date created on 08.09.2015
%

close all;
clear all;

fid = fopen('iris.data');
tline = fgets(fid);
data = [];
while ischar(tline) && length(tline) > 5
    pline = sscanf(tline, '%f,%f,%f,%f,%s');
    pline = pline';
    if strcmp(char(pline(5:end)), 'Iris-setosa')
      pline = [pline(1:4), 1];
    elseif strcmp(char(pline(5:end)), 'Iris-versicolor')
      pline = [pline(1:4), 2];
    elseif strcmp(char(pline(5:end)), 'Iris-virginica')
      pline = [pline(1:4), 3];
    else
      fprintf('"%s"\n', tline);
      error('Could not parse class: %s', pline(5:end));
    end
    data = [data; pline];
    tline = fgets(fid);
end
fclose(fid);

clear fid tline pline tmp;
x = data(:, 1:4);
t = data(:, 5);
x = bsxfun(@times, bsxfun(@minus, x, mean(x)), 1 ./ std(x));

save('iris.mat', 'x', 't');