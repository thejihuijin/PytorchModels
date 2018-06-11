%% 
clear; close all; clc
%% Read in Data
data = hdf5read('../EllipseGeneration/RandomLineEllipses15.hdf5','ellip/test_labels');
%% Evaluate
measurements = zeros(size(data));
recovered = zeros(size(data));
num_data = size(data,3);
n = 256;
timing = zeros(1,num_data);

% Loop through data
starttime = tic();
for i=1:num_data
    display(['Data ', num2str(i)])
    % Prepare data
    x = data(:,:,i);
    
    % Prepare A
    L = 15;
    [M,Mh,mh,mhi] = RandomLineMask(L,n);
    OMEGA = mhi;
    A = @(z) A_fhp(z, OMEGA);
    At = @(z) At_fhp(z, OMEGA, n);
    
    % take measurements
    y = A(x(:));

    % min l2 reconstruction (backprojection)
    xbp = At(y);
    Xbp = reshape(xbp,n,n);
    measurements(:,:,i) = Xbp;

    lambda = .1;

    K = @(x)grad(x);
    KS = @(x)-div(x);

    Amplitude = @(u)sqrt(sum(u.^2,3));
    F = @(u)lambda*sum(sum(Amplitude(u)));

    G_lm = @(x) 1/2*norm(y-A(x(:)))^2;

    Normalize = @(u)u./repmat( max(Amplitude(u),1e-10), [1 1 2] );
    ProxF = @(u,tau)repmat( perform_soft_thresholding(Amplitude(u),lambda*tau), [1 1 2]).*Normalize(u);
    ProxFS = compute_dual_prox(ProxF);

    makeb = @(tau,z) tau*xbp + z;
    makeH = @(tau) @(x) tau*At(A(x)) + x;
    ProxGlm = @(x,tau) reshape(cgd(makeH(tau),makeb(tau,x(:))),n,n);
    %ProxGlm = @(x,tau) reshape((eye(n*n)+tau*AtA)\(x(:)+tau*At(y)),n,n);

    options.report = @(x)G_lm(x) + F(K(x));
    options.niter = 1000;

%     options.method = 'fista';

    options.niter = 2000;
    admmstart = tic();
    [xAdmm,EAdmm] = perform_admm(Xbp, K,  KS, ProxFS, ProxGlm, options);
    timing(i) = toc(admmstart);
    recovered(:,:,i) = xAdmm;
    
end
toc(starttime)
%% record
% timing, recovered, measurements
i = 1;
figure;
subplot(131)
imagesc(measurements(:,:,i))
subplot(132)
imagesc(recovered(:,:,i))
title(['MSE = ', num2str(mean2((data(:,:,i)-recovered(:,:,i)).^2))])
subplot(133)
imagesc(data(:,:,i))
title('Ground Truth')

%% Single
% Prepare data
i=1;
n = 256;
x = data(:,:,i);

% Prepare A
L = 15;
[M,Mh,mh,mhi] = LineMask(L,n);
OMEGA = mhi;
A = @(z) A_fhp(z, OMEGA);
At = @(z) At_fhp(z, OMEGA, n);

% take measurements
y = A(x(:));

% min l2 reconstruction (backprojection)
xbp = At(y);
Xbp = reshape(xbp,n,n);

lambda = .1;

K = @(x)grad(x);
KS = @(x)-div(x);

Amplitude = @(u)sqrt(sum(u.^2,3));
F = @(u)lambda*sum(sum(Amplitude(u)));

G_lm = @(x) 1/2*norm(y-A(x(:)))^2;

Normalize = @(u)u./repmat( max(Amplitude(u),1e-10), [1 1 2] );
ProxF = @(u,tau)repmat( perform_soft_thresholding(Amplitude(u),lambda*tau), [1 1 2]).*Normalize(u);
ProxFS = compute_dual_prox(ProxF);

makeb = @(tau,z) tau*xbp + z;
makeH = @(tau) @(x) tau*At(A(x)) + x;
ProxGlm = @(x,tau) reshape(cgd(makeH(tau),makeb(tau,x(:))),n,n);
%ProxGlm = @(x,tau) reshape((eye(n*n)+tau*AtA)\(x(:)+tau*At(y)),n,n);

options.report = @(x)G_lm(x) + F(K(x));
options.niter = 1000;

options.method = 'fista';

options.niter = 2000;
admmstart = tic();
[xAdmm,EAdmm] = perform_admm(Xbp, K,  KS, ProxFS, ProxGlm, options);
%%
i = 1;
figure;
subplot(231)
imagesc(Xbp)
colorbar()
subplot(232)
imagesc(xAdmm)
colorbar()
title(['MSE = ', num2str(mean2((x-xAdmm).^2))])
subplot(233)
imagesc(x)
colorbar()
title('Ground Truth')
subplot(2,3,[4:6])
plot(EAdmm)

