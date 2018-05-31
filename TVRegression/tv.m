function [xAdmm] = tv(x,lambda)
[n,~] = size(x);
L = 22;
[M,Mh,mh,mhi] = LineMask(L,n);
OMEGA = mhi;
A = @(z) A_fhp(z, OMEGA);
At = @(z) At_fhp(z, OMEGA, n);

% measurements
y = A(x(:));

% min l2 reconstruction (backprojection)
xbp = At(y);
Xbp = reshape(xbp,n,n);

%lambda = 10;

K = @(x)grad(x);
KS = @(x)-div(x);

Amplitude = @(u)sqrt(sum(u.^2,3));
F = @(u)lambda*sum(sum(Amplitude(u)));
G = @(x)1/2*norm(y-x,'fro')^2;
G_lm = @(x) 1/2*norm(y-A(x(:)))^2;


Normalize = @(u)u./repmat( max(Amplitude(u),1e-10), [1 1 2] );
ProxF = @(u,tau)repmat( perform_soft_thresholding(Amplitude(u),lambda*tau), [1 1 2]).*Normalize(u);
ProxFS = compute_dual_prox(ProxF);

ProxG = @(x,tau)(x+tau*y)/(1+tau);
ProxGlm = @(x,tau)(x+tau*reshape(At(y),n,n))/(1+tau);

options.report = @(x)G_lm(x) + F(K(x));
options.niter = 500;

[xAdmm,EAdmm] = perform_admm(Xbp, K,  KS, ProxFS, ProxGlm, options);

figure;
subplot(131)
imageplot(x)
title('Original')
subplot(132)
imageplot(Xbp)
title('Start')
subplot(133)
imageplot(xAdmm);
title(['TV Reg lambda=',num2str( lambda)])
end

