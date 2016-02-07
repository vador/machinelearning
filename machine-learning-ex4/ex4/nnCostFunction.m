function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
 % y_i = eye(num_labels)(:,y(i))

Yb = eye(num_labels)(:,y) ;% Y(:,i) is y^(i) as binary vector  Yb is in columns
% Yb size is K*m 10*5000
a1= [ones(size(X, 1), 1) X];  % 5000 * 401 
z2 = (a1 * Theta1'); % 5000 * 25
a2 = [ones(size(z2, 1), 1) sigmoid(z2)]; % 5000 * 26
z3 = (a2 * Theta2'); % 5000 * 10
a3 = sigmoid(z3); % this is h_theta(x) size is m*K 
%[vp, p] = max(a3, [],2); % Prediction in p
h_theta = a3 ;

%J = -1/m*((Yb*log(h_theta))+((1-Yb)*log(1-h_theta))) + reg_factor ;
for i = 1:m
  tmp = (log(h_theta(i,:))*(Yb(:,i))+(log(1-h_theta(i,:)))*(1-Yb(:,i)))  ;
  J = J+tmp ;
end

J = -1/m * J ;

delta_3_full = (a3 - Yb') ; % size is m*K 5000*10

DELTA_1 = zeros(size(Theta1)) ; % u2 * (u1+1) 25*401
DELTA_2 = zeros(size(Theta2)) ; % K * (u2+1) 10*26

for i = 1:m
  delta_3 = delta_3_full(i,:)' ; % K*1 size 10*1
  %fprintf("delta_3 size %f, %f\n",size(delta_3,1), size(delta_3,2));

  delta_2 = (Theta2' * delta_3)(2:end) .* sigmoidGradient(z2(i,:)'); % u_2 * 1 size 25*1
  %fprintf("delta_2 size %f, %f\n",size(delta_2,1), size(delta_2,2));
  DELTA_2 = DELTA_2 + delta_3 * a2(i,:) ; % K * (u2) 10*26
  DELTA_1 = DELTA_1 + (delta_2 * a1(i,:)) ; % u2 * 25*401
end

Theta1_grad = 1/m*DELTA_1 ;
Theta2_grad = 1/m*DELTA_2 ;

% Regularization calculation

reg_fact = lambda / (2*m) * (sum(sumsq((Theta1(:,2:end) )))+ sum(sumsq(Theta2(:,2:end)))) ;
J = J + reg_fact ;

reg_grad1 = [zeros(size(Theta1,1),1) lambda/m*Theta1(:,2:end)] ;
reg_grad2 = [zeros(size(Theta2,1),1) lambda/m*Theta2(:,2:end)] ;
Theta1_grad = Theta1_grad + reg_grad1 ;
Theta2_grad = Theta2_grad + reg_grad2 ;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
