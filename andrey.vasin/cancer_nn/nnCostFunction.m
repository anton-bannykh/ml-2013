function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

X = [ones(m, 1) X];
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a = sigmoid(X * Theta1');

a = [ones(size(a, 1), 1) a];

h = sigmoid(a * Theta2');

yTmp = zeros(m, num_labels);

for i = 1:m
	yTmp(i, y(i, 1)) = 1;
end


for i = 1:m
	J = J - yTmp(i, :) * log(h(i, :))' - (1 - yTmp(i, :)) * log(1 - h(i, :)');
end

J = 1.0 / m * J;

Theta1new = Theta1;
Theta2new = Theta2;

for i = 1:size(Theta1, 1)
	Theta1new(i, 1) =  0;
end

for i = 1:size(Theta2, 1)
	Theta2new(i, 1) =  0;
end


J = J + lambda / (2 * m) * (sum(sum(Theta1new .^ 2)) + sum(sum(Theta2new .^ 2)));

for i = 1:m
	d3 = (h(i, :) - yTmp(i, :))';
    tmp = Theta2' * d3;
	d2 = tmp(2:end) .* sigmoidGradient(X(i, :) * Theta1')';
	Theta2_grad = Theta2_grad +  d3 * a(i, :);
	Theta1_grad = Theta1_grad +  d2 * X(i, :);
end

Theta1Tmp = [zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2Tmp = [zeros(size(Theta2,1),1) Theta2(:, 2:end)];

Theta1_grad = 1.0 / m .* (Theta1_grad + lambda .* Theta1Tmp);
Theta2_grad = 1.0 / m .* (Theta2_grad + lambda .* Theta2Tmp);


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
