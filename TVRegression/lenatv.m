%%
clear;clc;close all;
%%
addpath('./toolbox_optim','./toolbox_optim/toolbox')
%% Load Lena
n = 256;
x = load_image('lena',n*2);
x = rescale(crop(x,n));
%y = y + randn(n)*.06;

clf;
imageplot(clamp(x));
%% Load Ellipse
load('../data/preproc_x20_ellipse_fullfbp.mat','lab_n')
x = lab_n(:,:,1,1);
clear('lab_n');
%%
n = 128;
x = imresize(x,[n,n]);
clf;
imageplot(x);
%% Set up AtA matrix
L = 15;
[M,Mh,mh,mhi] = LineMask(L,n);
OMEGA = mhi;
A = @(z) A_fhp(z, OMEGA);
At = @(z) At_fhp(z, OMEGA, n);
%%
F = dftmtx(n);
fft2d = kron(F,F)/n;
clear F;

mask = M;
m = length(find(mask));
sample_mat = zeros(numel(mask),numel(mask));
idx = sub2ind(size(sample_mat), find(mask), find(mask));
sample_mat(idx) = 1;

clear idx m mask

% AtA = real(fft2d'*sample_mat*fft2d);
%AtAinv = inv(AtA);
clear fft2d sample_mat
%% Set up optimization
% measurements
y = A(x(:));

% min l2 reconstruction (backprojection)
xbp = At(y);
Xbp = reshape(xbp,n,n);

lambda = 10;

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

GradGS = @(x)x+y;
GradGS_lm = @(x)reshape(AtAinv*(x(:)+xbp),n,n);
L = 8;
options.method = 'fista';

%%
%[xFista,EFista] = perform_fb_strongly(Xbp, K, KS, GradGS_lm, ProxFS, L, options);
options.niter = 10000;
[xAdmm,EAdmm] = perform_admm(Xbp, K,  KS, ProxFS, ProxGlm, options);
%%
G = @(x)1/2*norm(Xbp-x,'fro')^2;
ProxG = @(x,tau)(x+tau*Xbp)/(1+tau);
options.report = @(x)G(x) + F(K(x));
[xAdmm2,EAdmm2] = perform_admm(Xbp, K,  KS, ProxFS, ProxG, options);

%% Continue training
%load('Admm.mat')
%options.niter = 200;
%[xAdmm_cont,EAdmm2_cont] = perform_admm(xAdmm, K,  KS, ProxFS, ProxGlm, options);
%%
figure;
subplot(221)
imageplot(x)
title('Original')
subplot(222)
imageplot(Xbp)
title('Start')
subplot(223)
imageplot(xAdmm);
title('TV Reg')
%subplot(224)
%imageplot(xAdmm_cont);
%title('Denoising')
%%
figure; hold all
plot(EAdmm);
% plot(EAdmm2);
% legend('TV','Denoising');
%%
save('Admm.mat','xAdmm','xAdmm_cont')
%%
xtv = tv(x,10);