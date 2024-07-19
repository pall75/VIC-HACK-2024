import nibabel as nib
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from scipy.optimize import least_squares
from scipy.special import ive  # Modified Bessel function of the first kind
from nibabel.viewers import OrthoSlicer3D as ov
import matplotlib.pyplot as plt
from scipy.io import savemat

print('It gots here')

# Read NIfTI files and convert to double
# dwi_synth_1 = nib.load(r'C:\Users\c1668701\OneDrive - Cardiff University\Documents\GitHub\VIC-HACK-2024_2\NaNN\syn_data\dwi_synth_1.nii.gz').get_fdata().astype(np.float64)
# dwi_synth_2 = nib.load(r'C:\Users\c1668701\OneDrive - Cardiff University\Documents\GitHub\VIC-HACK-2024_2\NaNN\syn_data\dwi_synth_2.nii.gz').get_fdata().astype(np.float64)
dwi_synth_1 = nib.load(r'/home/c1668701/VIC-HACK-2024/NaNN/syn_data/dwi_synth_1.nii.gz').get_fdata().astype(np.float64)
dwi_synth_2 = nib.load(r'/home/c1668701/VIC-HACK-2024/NaNN/syn_data/dwi_synth_2.nii.gz').get_fdata().astype(np.float64)

# Concatenate along the fourth dimension
dwi_synth = np.concatenate((dwi_synth_1[..., np.newaxis], dwi_synth_2[..., np.newaxis]), axis=3)

# Extract specific slices and volumes
B0s = dwi_synth[43:86, 43:86, 33:34, :9]

# Calculate the mean and sum of squares
m_1_data = np.mean(B0s, axis=3)
m_2_data = (1 / B0s.shape[3]) * np.sum(B0s ** 2, axis=3)

# Create brain mask
# brain_mask = nib.load(r'C:\Users\c1668701\OneDrive - Cardiff University\Documents\GitHub\VIC-HACK-2024_2\NaNN\syn_data/dwi_synth_mask.nii.gz').get_fdata().astype(np.float64)
brain_mask = nib.load(r'/home/c1668701/VIC-HACK-2024/NaNN/syn_data/dwi_synth_mask.nii.gz').get_fdata().astype(np.float64)
brain_mask = brain_mask[43:86,43:86, 33:34]

# Find indices of brain mask
ind_brain = np.nonzero(brain_mask)

# Function to calculate moments of the Rician distribution
def moments_rician_vxls(p,m1,m2):
    sigma = p[0]

    nu2 = p[1] ** 2
    sigma2 = sigma ** 2

    x = -nu2 / (2 * sigma2)
    I0_mxp5 = ive(0, -x / 2)
    I1_mxp5 = ive(1, -x / 2)

    L_0p5_x = np.exp(x / 2) * ((1 - x) * I0_mxp5 - x * I1_mxp5)
    mus_rician_1 = sigma * np.sqrt(np.pi / 2) * L_0p5_x
    mus_rician_2 = 2 * sigma2 + nu2

    return (mus_rician_1-m1, mus_rician_2-m2)

# sol = least_squares(moments_rician,x0.flatten(),verbose=2, bounds=(-1, 10))

m_1_data_brain = m_1_data[ind_brain]
m_2_data_brain = m_2_data[ind_brain]
sigma_est=np.zeros(m_1_data_brain.shape)
nu_est=np.zeros(m_1_data_brain.shape)
m_1_est=np.zeros(m_1_data_brain.shape)
m_2_est=np.zeros(m_1_data_brain.shape)
for i_v in range(len(m_1_data_brain)):
    sol = least_squares(moments_rician_vxls,(0.17,2.6), bounds=([0.1, 0], [0.4, 6]), args=(np.take(m_1_data_brain,i_v), np.take(m_2_data_brain,i_v)))
    sigma_est[i_v], nu_est[i_v] = sol.x
    m_1_est[i_v], m_2_est[i_v] = sol.fun
    print(i_v)

mdic = {"sigma_est": sigma_est, "nu_est": nu_est, "brain_mask": brain_mask,
        "m_1_data": m_1_data, "m_2_data": m_2_data,"m_1_est": m_1_est, "m_2_est": m_2_est}
savemat('/home/c1668701/VIC-HACK-2024/NaNN/syn_data/sigma_est_MoM.mat',mdic)
print(sol)

