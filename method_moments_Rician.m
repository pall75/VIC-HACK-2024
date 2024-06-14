clear all
close all
clc
addpath(genpath('C:\Users\c1668701\OneDrive - Cardiff University\Documents\GitHub\VIC-HACK-2024_2\NaNN'))

dwi_synth_1=double(niftiread('dwi_synth_1.nii.gz'));
dwi_synth_2=double(niftiread('dwi_synth_2.nii.gz'));
dwi_synth=cat(4,dwi_synth_1,dwi_synth_2);

B0s=dwi_synth(43:86,43:86,29:30,1:9);

m_1_data=mean(B0s,4);
m_2_data=1/9*sum(B0s.^2,4);

brain_mask=mean(B0s,4)>0.5;
SE = strel("cuboid",[5 5 1]);
brain_mask=imclose(imopen(brain_mask,SE),SE);

ind_brain=find(brain_mask);
y_data=[m_1_data(ind_brain) m_2_data(ind_brain)];

x0=ones(length(ind_brain),2);

options = optimoptions('lsqcurvefit','Display','iter');
pars_hat=lsqcurvefit(@moments_rician,x0,x0,y_data,[],[],options);

function mus_rician = moments_rician(par,xdata)
%https://en.wikipedia.org/wiki/Rice_distribution
    sigma=par(:,1);
    nu=par(:,2);
        
    nu2=nu.^2;
    sigma2=sigma.^2;

    x=-nu2./(2*sigma2);
    I0_mxp5=besseli(0,-x/2);
    I1_mxp5=besseli(1,-x/2);    
    
    L_0p5_x=exp(x/2).*((1-x).*I0_mxp5-x.*I1_mxp5);
    mus_rician(:,1)=sigma.*sqrt(pi/2).*L_0p5_x;
    mus_rician(:,2)= 2*sigma2+nu2;
end