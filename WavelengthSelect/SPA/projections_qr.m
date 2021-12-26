function chain = projections_qr(X,k,M)

% Projections routine for the Successive Projections Algorithm using the
% built-in QR function of Matlab
%
% chain = projections(X,k,M)
%
% X --> Matrix of predictor variables (# objects N x # variables K)
% k --> Index of the initial column for the projection operations
% M --> Number of variables to include in the chain
%
% chain --> Index set of the variables resulting from the projection operations

X_projected = X;

norms = sum(X_projected.^2);    % Square norm of each column vector
norm_max = max(norms); % Norm of the "largest" column vector

X_projected(:,k) = X_projected(:,k)*2*norm_max/norms(k); % Scales the kth column so that it becomes the "largest" column

[dummy1,dummy2,order] = qr(X_projected,0); 
chain = order(1:M)';