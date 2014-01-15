function svmCancer(testPart)
[X, y] = loadData('wdbc.data');
X = X.';
testNum = int32(length(y)*testPart);
learnMask = randsample(1:length(y), testNum);
testMask = zeros(1, length(y));
testMask(learnMask) = learnMask;
testMask = find(((1:length(y))-testMask) > 0);
%C = getC(X(learnMask,:), y(learnMask), 5);
C=0.1
[w, b] = svmqpTrain(X(learnMask,:), y(learnMask), C);
f = @(ax, aw) sign(aw.'*ax) + b;
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

function [w, b] = svmqpTrain(X, y, C)
alpha = quadprog((X*X.').*(y.'*y), -ones(length(y), 1), [eye(length(y)).' -eye(length(y)).'].', [C*ones(length(y), 1).' zeros(length(y), 1).'].', y, 0);
w = 1:length(X(1, :));
for i = 1:length(y)
    if alpha(i) > 0
        w = w + y(i).*alpha(i).*X(i,:);
    end
end
b = 0;
for i = 1:length(alpha)
    if alpha(i) < C
        b = y(i) - w*X(i,:).';
        break;
    end
end

end

function bestC = getC(X, y, parts)
bestC = 0;
bestError = inf;
st = floor(length(y)/parts);
for C = (1:100)*0.001;
    for i = (0:parts-1);
        lMask = [1:i*st ((i+1)*st+1):length(y)];
        tMask = i*st+1:(i+1)*st;
        [w, b] = svmqpTrain(X(lMask,:), y(lMask), C);
        f = @(ax, aw) sign(aw.'*ax) + b;
        errors(i+1) = length(find(f(X(tMask,:).', w.') ~= y(tMask)))/st;
    end
    curError = mean(errors);
    if curError < bestError
        bestError = curError;
        bestC = C;
    end
end
end