function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

tmp = (X * Theta' - Y);
tmp = (tmp .^ 2).* R;

J = sum(tmp(:)) / 2.0;

J = J + lambda / 2.0 * (sum(sum((Theta .^2))) + sum(sum((X .^2))));

for i = 1:size(X,1)
	idx = find(R(i, :) == 1);
	ThetaTmp = Theta(idx, :);
	YTmp = Y(i, idx);
	tmp = X(i,:) * ThetaTmp' - YTmp;
	X_grad(i,:) = (X(i,:) * ThetaTmp' - YTmp) * ThetaTmp + lambda * X(i,:);
end

for i = 1:size(Theta,1)
	idx = find(R(:, i) == 1);
	XTmp = X(idx, :);
	YTmp = Y(idx, i);
	Theta_grad(i,:) = (XTmp * (Theta(i, :))' - YTmp)' * XTmp + lambda * Theta(i, :);
end

grad = [X_grad(:); Theta_grad(:)];

end
