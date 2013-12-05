function [X, y] = loadData(fileName)
X = importdata(fileName, ',');
yc = cell2mat(X.textdata(:,2));
y(find(yc == 'B')) = 1;
y(find(yc == 'M')) = 2;
y = y';
X = (X.data).';
X = X';
end
