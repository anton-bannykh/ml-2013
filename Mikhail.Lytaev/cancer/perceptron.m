function perceptron(testPart)
    [X, y] = loadData('wdbc.data');
    testNum = int32(length(y)*testPart);
    learnMask = randsample(1:length(y), testNum);
    testMask = zeros(1, length(y));
    testMask(learnMask) = learnMask;
    testMask = find(((1:length(y))-testMask) > 0);
    [w, ni] = train(X(:, learnMask), y(learnMask), 100000);
    f = @(ax, aw) sign(aw.'*ax);
    classRes = f(X(:,testMask), w);
    y = y(testMask);
    tp = length(find(classRes == y & y == 1)) %true positive
    tn = length(find(classRes == y & y == -1)) %true negative
    fp = length(find(classRes ~= y & y == 1)) %false positive
    fn = length(find(classRes ~= y & y == -1)) %false negative
    precision = tp/(tp+fp);
    recall = tp/(tp + fn);
    disp(['precision = ', num2str(precision), ' recall = ', num2str(recall), ' ni = ', num2str(ni)])
end

function [X, y] = loadData(fileName)
X = importdata(fileName, ',');
yc = cell2mat(X.textdata(:,2));
y(find(yc == 'M')) = -1;
y(find(yc == 'B')) = 1;
X = (X.data).';
end

function [w, i] = train(X, y, maxIter)
w = X(:,1).*0;
f = @(ax, aw) sign(aw.'*ax);
for i=1:maxIter
    wrong = find(f(X, w) ~= y);
    if isempty(wrong) ,return ,end
    t = randsample(wrong, 1);
    w = w + y(t)*X(:,t);
end
end