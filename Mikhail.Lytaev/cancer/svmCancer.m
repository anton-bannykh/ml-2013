function svmCancer(testPart)
    [X, y] = loadData('wdbc.data');
    X = X.';
    testNum = int32(length(y)*testPart);
    learnMask = randsample(1:length(y), testNum);
    testMask = zeros(1, length(y));
    testMask(learnMask) = learnMask;
    testMask = find(((1:length(y))-testMask) > 0);
    C = getC(X(learnMask,:), y(learnMask), 5);
    sst = svmtrain(X(testMask,:), y(testMask), 'boxconstraint', C);
    classRes = svmclassify(sst, X(testMask,:));
    error = length(find(classRes ~= y(testMask).'))/length(y);
    disp(['error = ', num2str(error), ' C = ', num2str(C)])
end

function [X, y] = loadData(fileName)
X = importdata(fileName, ',');
yc = cell2mat(X.textdata(:,2));
y(find(yc == 'M')) = -1;
y(find(yc == 'B')) = 1;
X = (X.data).';
end

function bestC = getC(X, y, parts)
bestC = 0;
bestError = inf;
st = floor(length(y)/parts);
for C = (1:9).*0.1;
    for i = (0:parts-1);
        lMask = [1:i*st ((i+1)*st+1):length(y)];
        tMask = i*st+1:(i+1)*st;
        sst = svmtrain(X(lMask,:), y(lMask), 'boxconstraint', C);
        errors(i+1) = length(find(svmclassify(sst, X(tMask,:)) ~= y(tMask).'))/st;
    end
    curError = mean(errors);
    if curError < bestError
        bestError = curError;
        bestC = C;
    end
end
end