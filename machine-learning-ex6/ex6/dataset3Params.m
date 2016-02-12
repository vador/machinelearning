function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
Cvalues = [0.01 ; 0.03; 0.1; 0.3; 1; 3; 10; 30];
Sigmavalues = [0.01 ; 0.03; 0.1; 0.3; 1; 3; 10; 30];

%Cvalues = [0.01 ; 0.1; 1; 10];
%Sigmavalues = [0.01 ; 0.1; 1; 10];

minpreerror = 9999999;
minC = -1 ;
minSig = -1 ;

for i = 1:length(Cvalues)
  C = Cvalues(i) 
  for j = 1:length(Sigmavalues)
    sigma = Sigmavalues(j)
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval) ;

    prederror = mean(double(predictions ~= yval));
    if (prederror < minpreerror)
      minpreerror = prederror ;
      minC = C ;
      minSig = sigma ;
    end
  end
end

C = minC 
sigma = minSig 


% =========================================================================

end
