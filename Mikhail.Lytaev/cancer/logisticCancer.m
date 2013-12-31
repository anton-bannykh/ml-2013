function logisticCancer(testPart)
[X, y] = loadData('wdbc.data');
X = X.';
testNum = int32(length(y)*testPart);
learnMask = randsample(1:length(y), testNum);
testMask = zeros(1, length(y));
testMask(learnMask) = learnMask;
testMask = find(((1:length(y))-testMask) > 0);
%lambda = getLambda(X(learnMask,:), y(learnMask), 5)
lambda =0.05;
w = logisticTrain(X(learnMask,:), y(learnMask), lambda);
f = @(ax, aw) sign(aw.'*ax);
classRes = f(X(testMask,:).', w.');
y = y(testMask);
tp = length(find(classRes == y & y == 1)) %true positive
tn = length(find(classRes == y & y == -1)) %true negative
fp = length(find(classRes ~= y & y == 1)) %false positive
fn = length(find(classRes ~= y & y == -1)) %false negative
precision = tp/(tp+fp);
recall = tp/(tp + fn);
f1 = 2*(precision*recall)/(precision+recall);
disp(['precision = ', num2str(precision), ' recall = ', num2str(recall), ' f1 = ', num2str(f1)])
end

function [X, y] = loadData(fileName)
X = importdata(fileName, ',');
yc = cell2mat(X.textdata(:,2));
y(find(yc == 'M')) = 1;
y(find(yc == 'B')) = -1;
X = (X.data).';
end

function w = logisticTrain(X, y, lambda)
g = @(z) 1./(1+exp(-z));
f = @(w) -sum(y.*log(1./(g(w*X.')))+(1-y).*log(1./(g(-w*X.'))))+1./2.*lambda.*sum(w.*w);
w = fminsearch(f, zeros(1,length(X(1,:))));
end

function bestLambda = getLambda(X, y, parts)
bestLambda = 0;
bestError = inf;
st = floor(length(y)/parts);
for C = (1:100)*0.001;
    for i = (0:parts-1);
        lMask = [1:i*st ((i+1)*st+1):length(y)];
        tMask = i*st+1:(i+1)*st;
        w = logisticTrain(X(lMask,:), y(lMask), C);
        f = @(ax, aw) sign(aw.'*ax);
        errors(i+1) = length(find(f(X(tMask,:).', w.') ~= y(tMask)))/st;
    end
    curError = mean(errors);
    if curError < bestError
        bestError = curError;
        bestC = C;
    end
end
end