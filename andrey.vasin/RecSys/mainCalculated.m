trainFiles = ['movielensfold1.txt'; 'movielensfold2.txt'; 'movielensfold3.txt'; 'movielensfold4.txt'; 'movielensfold5.txt'];
ansFiles = ['movielensfold1ans.txt'; 'movielensfold2ans.txt'; 'movielensfold3ans.txt'; 'movielensfold4ans.txt'; 'movielensfold5ans.txt'];
bestLambda = 0.08;
bestNum_features = 3;

for q = 1:size(trainFiles)
	[num_movies, num_users, trainSet, testSet] = loadData(trainFiles(q, :), ansFiles(q, :));
	Ytrain = zeros(num_movies, num_users);
	for i = 1:size(trainSet, 1)
		Ytrain(trainSet(i, 2) + 1, trainSet(i, 1) + 1) = trainSet(i, 3);
	end
	
	Ytest = zeros(num_movies, num_users);
	for i = 1:size(testSet, 1)
		Ytest(testSet(i, 2) + 1, testSet(i, 1) + 1) = testSet(i, 3);
	end
	
	Rtrain = [(Ytrain ~= 0)];
	
	Rtest = [(Ytest ~= 0)];
	
	X = randn(num_movies, bestNum_features);
	Theta = randn(num_users, bestNum_features);
	
	initial_parameters = [X(:); Theta(:)];

	options = optimset('GradObj', 'on', 'MaxIter', 100);

	theta = fmincg (@(t)(cofiCostFunc(t, Ytrain, Rtrain, num_users, num_movies, ...
									bestNum_features, bestLambda)), ...
					initial_parameters, options);
					
	error = cofiCostFunc(theta, Ytest, Rtest, num_users, num_movies, ...
									bestNum_features, 0);
	error = 2.0 * error / size(testSet,1);
	disp(['error = ', num2str(error)]);
end