clc 
close all
clear all
%%
load('RG_256_by_1024.mat')
%%
waveletType = 'Daubechies';
waveletPar = 8;
waveletLevel = 3;
QMF = MakeONFilter(waveletType,waveletPar);

sparse_img = zeros(32,32,3);
msr_img = zeros(16,16,3);
final_sparse = zeros(10000,3072);
final_msr = zeros(10000,768);

load('../../data/cifar-10-batches-mat/data_batch_1.mat')


%%
for ii = 1 : 10000
    
    waitbar(ii/10000)

    tmp_img = im2double(imrotate(reshape(data(ii,:),[32,32,3]),-90));

    waveImg1 = FWT2_PO(tmp_img(:,:,1),waveletLevel,QMF);
    waveImg2 = FWT2_PO(tmp_img(:,:,2),waveletLevel,QMF);
    waveImg3 = FWT2_PO(tmp_img(:,:,3),waveletLevel,QMF);

    sorted_WT_coeffs1 = sort(abs(waveImg1(:)),'descend');
    sorted_WT_coeffs2 = sort(abs(waveImg2(:)),'descend');
    sorted_WT_coeffs3 = sort(abs(waveImg3(:)),'descend');

    cutoff_thresh = 0.25;

    cutoff_coeff1 = sorted_WT_coeffs1(floor(cutoff_thresh*length(sorted_WT_coeffs1)));
    cutoff_coeff2 = sorted_WT_coeffs2(floor(cutoff_thresh*length(sorted_WT_coeffs2)));
    cutoff_coeff3 = sorted_WT_coeffs3(floor(cutoff_thresh*length(sorted_WT_coeffs3)));

    waveImg1(abs(waveImg1) < cutoff_coeff1) = 0;
    waveImg2(abs(waveImg2) < cutoff_coeff2) = 0;
    waveImg3(abs(waveImg3) < cutoff_coeff3) = 0;

    Img1 = IWT2_PO(waveImg1,waveletLevel,QMF);
    sparse_img(:,:,1) = Img1;
    Img2 = IWT2_PO(waveImg2,waveletLevel,QMF);
    sparse_img(:,:,2) = Img2;
    Img3 = IWT2_PO(waveImg3,waveletLevel,QMF);
    sparse_img(:,:,3) = Img3;
    
    rp_img1 = reshape(A * reshape(tmp_img(:,:,1),[1024, 1]),[16,16]);
    msr_img(:,:,1) = rp_img1;
    rp_img2 = reshape(A * reshape(tmp_img(:,:,2),[1024, 1]),[16,16]);
    msr_img(:,:,2) = rp_img2;
    rp_img3 = reshape(A * reshape(tmp_img(:,:,3),[1024, 1]),[16,16]);
    msr_img(:,:,3) = rp_img3;
    
    
    
    final_sparse(ii,:) = reshape(sparse_img,[1,3072]);
    final_msr(ii,:) = reshape(msr_img ,[1,768]);
end

save('CIFAR1_sparse_train.mat','final_sparse')
save('CIFAR1_msr_train.mat','final_msr')

