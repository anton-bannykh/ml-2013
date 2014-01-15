function [num_movies, num_users, trainSet, testSet] = loadData(fileName1, fileName2)
fid1 = fopen(fileName1, 'rb');
fid2 = fopen(fileName2, 'rb');
input = fscanf (fid1,'%d %d %d %d %d\n', [1 5]);
num_users = input(2);
num_movies = input(3);
trainSize = input(4);
testSize = input(5);
trainSet = zeros(trainSize, 3);
for i = 1:trainSize
	trainSet(i,:) = fscanf(fid1, '%d %d %d\n', [1 3]);
end
testSet = zeros(testSize, 3);
for i = 1:testSize
	testSet(i,1:2) = fscanf(fid1, '%d %d\n', [1 2]);
	testSet(i, 3) = fscanf(fid2, '%d\n', [1 1]);
end
end
