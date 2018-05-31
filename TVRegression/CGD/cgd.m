function [finalx, residuals] = cgd(H,b)
%CGD 
% H - function. For simple linear case should be an application of A
% followed by A^T
% b - vector, assumed to be of size N x 1

N = numel(b);
epsilon = 10^-4;
residuals = [];

% Initialize
x = zeros(N,1);
r = b - H(x);
d = r;

for k = 0:N-1
    alpha = r.'*r/(d.'*H(d));
    x = x + alpha*d;
    nextr = r - alpha*H(d);
    beta = nextr.'*nextr/(r.'*r);
    r = nextr;
    d = r + beta*d;
    
    % check residual size
    error = sqrt(r.'*r);
    if error < epsilon
        break;
    else
        residuals = [residuals error];
    end
end
finalx = x;
end

