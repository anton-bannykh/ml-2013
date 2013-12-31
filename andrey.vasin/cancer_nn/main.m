clear ; close all; clc

[X, y] = loadData('wdbc.data');

input_layer_size  = size(X, 2);
testPart = 0.2;
num_labels = 2;  
parts = 5; 

testNum = int32(length(y)*testPart);
testMask = randsample(1:length(y), testNum);
learnMask = zeros(1, length(y));
learnMask(testMask) = testMask;
learnMask = find(((1:length(y))-learnMask) > 0);

trainSetX = X(learnMask, :);
trainSety = y(learnMask);
checkSetX = X(testMask, :);
checkSety = y(testMask);

bestError = 2;
bestC = 0;
bestHLSize = input_layer_size;

st = floor(length(trainSety)/parts);

for hidden_layer_size = 1:25
    disp(['Layer size = ', num2str(hidden_layer_size)])
	for i = 1:10
		lambda = (2.0 ^ i) / 100.0;
        disp(['Lambda = ', num2str(lambda)])
		
		curError = 0.0;
		
		for j = (0:parts-1)
			lMask = [1:j*st ((j+1)*st+1):length(trainSety)];
			tMask = j*st+1:(j+1)*st;
		
			initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
			initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
			initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
			
			costFunction = @(p) nnCostFunction(p, ...
									   input_layer_size, ...
									   hidden_layer_size, ...
									   num_labels, trainSetX(lMask, :), trainSety(lMask), lambda);
									   
			options = optimset('MaxIter', 50);
			[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
			
			Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
					 hidden_layer_size, (input_layer_size + 1));
			Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
					 num_labels, (hidden_layer_size + 1));
					 
			pred = predict(Theta1, Theta2, trainSetX(tMask, :));
			
			curError = curError + mean(double(pred ~= trainSety(tMask)));
        end
		
		curError = curError / parts;
		
		if curError < bestError
			bestError = curError;
			bestC = lambda;
			bestHLSize = hidden_layer_size;
        end
    end
end

hidden_layer_size = bestHLSize;
lambda = bestC;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

costFunction = @(p) nnCostFunction(p, ...
						   input_layer_size, ...
						   hidden_layer_size, ...
						   num_labels, trainSetX, trainSety, lambda);

options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		 num_labels, (hidden_layer_size + 1));
		 
pred = predict(Theta1, Theta2, checkSetX);

tp = length(find(pred == checkSety & checkSety == 2)) %true positive
tn = length(find(pred == checkSety & checkSety == 1)) %true negative
fp = length(find(pred ~= checkSety & checkSety == 2)) %false positive
fn = length(find(pred ~= checkSety & checkSety == 1)) %false negative
precision = tp/(tp+fp);
recall = tp/(tp + fn);
f1 = 2*(precision*recall)/(precision+recall);
disp(['hidden layer size = ', num2str(hidden_layer_size),' constant = ', num2str(lambda), ' precision = ', num2str(precision), ' recall = ', num2str(recall), ' f1 = ', num2str(f1)])