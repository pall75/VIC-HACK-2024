clear all
close all
clc
addpath(genpath('C:\Users\c1668701\OneDrive - Cardiff University\Documents\GitHub\VIC-HACK-2024_2\NaNN'))
addpath(genpath('/home/c1668701/VIC-HACK-2024/NaNN/syn_data'))


%% MLE
dwi_synth_1=double(niftiread('dwi_synth_1.nii.gz'));
dwi_synth_2=double(niftiread('dwi_synth_2.nii.gz'));
dwi_synth=cat(4,dwi_synth_1,dwi_synth_2);

B0s=dwi_synth(:,:,33:34,1:9);

brain_mask=double(niftiread('/home/c1668701/VIC-HACK-2024/NaNN/syn_data/dwi_synth_mask.nii.gz'));
brain_mask=brain_mask(:,:,33:34);
ind_brain=find(brain_mask);

B0s_brainList=zeros([prod(size(B0s,1:3)),9]);
for i_b=1:9
    B0s_brainList(:,i_b)=reshape(B0s(:,:,:,i_b),1,[]);
end

cube_str=strel('sphere',1);
[x_incr,y_incr,z_incr]=ind2sub([3,3,3],find(cube_str.Neighborhood));
x_incr=x_incr-2;
y_incr=y_incr-2;
z_incr=z_incr-2;
cube_str=strel('disk',1);
[x_incr,y_incr]=ind2sub([3,3],find(cube_str.Neighborhood));
x_incr=x_incr-2;
y_incr=y_incr-2;
z_incr=zeros(size(y_incr));

sigma_mle=zeros(size(brain_mask));
nu_mle=zeros(size(brain_mask));
pars_hat_list=zeros([length(ind_brain),2]);
for i_v=1:length(ind_brain)
    [x_v,y_v,z_v]=ind2sub(size(brain_mask),ind_brain(i_v));
    cube_ind=[x_v,y_v,z_v]+[x_incr,y_incr,z_incr];
    cube_ind(any(cube_ind<1 | cube_ind>size(brain_mask),2),:)=[];
    i_v_neighb=sub2ind(size(brain_mask),cube_ind(:,1),cube_ind(:,2),cube_ind(:,3));
    pars_hat_list(i_v,:)=mle(reshape(B0s_brainList(i_v_neighb,:),1,[]),'Distribution','rician');
    disp(i_v)
end
    % s: noncentrality parameter	σ: scale parameter
nu_mle(ind_brain)=pars_hat_list(:,1);
sigma_mle(ind_brain)=pars_hat_list(:,2);

sigma_gt=double(niftiread('sigma_map_gt.nii.gz'));
sigma_gt=sigma_gt(:,:,34);

figure;
subplot 221
imagesc(nu_mle(:,:,1));colorbar;title('nu')
subplot 223
imagesc(sigma_gt(:,:,1));colorbar;title('sigma gt')
caxis([0.15,.2]);c_axis=caxis;
subplot 222
imagesc(sigma_mle(:,:,1));colorbar;title('sigma est')
caxis(c_axis);
subplot 224
imagesc(sigma_mle(:,:,1)-sigma_gt(:,:,1));colorbar;title('sigma est - sigma gt')
caxis([-0.1,0.1]);

 return
 
%% From Python
load /home/c1668701/VIC-HACK-2024/NaNN/syn_data/sigma_est_MoM.mat
ind_brain=find(brain_mask);
nu_map=zeros(size(brain_mask));
sigma_map=zeros(size(brain_mask));
sigma_map(ind_brain)=sigma_est;
nu_map(ind_brain)=nu_est;

sigma_gt=double(niftiread('sigma_map_gt.nii.gz'));
sigma_gt=sigma_gt(43:86,43:86,28);

figure
subplot 221;imagesc(sigma_map)
subplot 222;imagesc(nu_map)
subplot 223;imagesc(sigma_gt)
subplot 224;imagesc(sigma_map-sigma_gt)