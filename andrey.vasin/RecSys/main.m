trainFiles = ['movielensfold1.txt'; 'movielensfold2.txt'; 'movielensfold3.txt'; 'movielensfold4.txt'; 'movielensfold5.txt'];
ansFiles = ['movielensfold1ans.txt'; 'movielensfold2ans.txt'; 'movielensfold3ans.txt'; 'movielensfold4ans.txt'; 'movielensfold5ans.txt'];
parts = 5;

for q = 1:size(trainFiles)
	[num_movies, num_users, trainSet, testSet] = loadData(trainFiles(q, :), ansFiles(q, :));
	bestError = inf;
	bestNum_features = -1;
	bestLambda = 0;
	st = floor(length(trainSet)/parts);
	for num_features = 1:10
		disp(['num_features = ', num2str(num_features)])
		for i = 1:10
			disp(['lambda = 2^', num2str(i)])
			lambda = (2.0 ^ i) / 100.0;
			error = 0.0;
			for j = (0:parts-1)
				lMask = [1:j*st ((j+1)*st+1):length(trainSet)];
				tMask = j*st+1:(j+1)*st;
				
				trainSetCV = trainSet(lMask, :);
				testSetCV = trainSet(tMask, :);
			
				Ytrain = zeros(num_movies, num_users);
				for z = 1:size(trainSetCV, 1)
					Ytrain(trainSetCV(z, 2) + 1, trainSetCV(z, 1) + 1) = trainSetCV(z, 3);
				end
				
				Ytest = zeros(num_movies, num_users);
				for z = 1:size(testSetCV, 1)
					Ytest(testSetCV(z, 2) + 1, testSetCV(z, 1) + 1) = testSetCV(z, 3);
				end
				
				Rtrain = [(Ytrain ~= 0)];
				
				Rtest = [(Ytest ~= 0)];
				
				X = randn(num_movies, num_features);
				Theta = randn(num_users, num_features);
				
				initial_parameters = [X(:); Theta(:)];

				options = optimset('GradObj', 'on', 'MaxIter', 50);

				theta = fmincg (@(t)(cofiCostFunc(t, Ytrain, Rtrain, num_users, num_movies, ...
												num_features, lambda)), ...
								initial_parameters, options);
								
				curError = cofiCostFunc(theta, Ytest, Rtest, num_users, num_movies, ...
												num_features, 0);
				error = error + curError;
			end
			error = error / parts;
			if bestError > error
				bestError = error;
				bestNum_features = num_features;
				bestLambda = lambda;
			end
		end
	end
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
	disp(['error = ', num2str(error), '; reg constant = ', num2str(bestLambda), '; number of features = ', num2str(bestNum_features)]);
end