function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X m*n
% y m*1
% theta :n * 1
% grad : n*1
% X*theta : m*1

reg_factor = lambda / (2*m) * sum((theta(2:end)).^2) ;
J = 1/2 / m * sum((X*theta-y) .^2) + reg_factor ;
%z = X*theta- y ;
%J = 1/(2*m)*sum(z.^2) ;

gradreg_fact = zeros(size(theta)) ;
gradreg_fact = lambda / m * theta ;
gradreg_fact(1) = 0 ;
grad =  1/m *X'*(X*theta -y) + gradreg_fact ;









% =========================================================================

grad = grad(:);

end
