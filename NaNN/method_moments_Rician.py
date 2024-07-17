import nibabel as nib
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from scipy.optimize import least_squares
from scipy.special import ive  # Modified Bessel function of the first kind


print('It gots here')

# Read NIfTI files and convert to double
dwi_synth_1 = nib.load(r'C:\Users\c1668701\OneDrive - Cardiff University\Documents\GitHub\VIC-HACK-2024_2\NaNN\syn_data\dwi_synth_1.nii.gz').get_fdata().astype(np.float64)
dwi_synth_2 = nib.load(r'C:\Users\c1668701\OneDrive - Cardiff University\Documents\GitHub\VIC-HACK-2024_2\NaNN\syn_data\dwi_synth_2.nii.gz').get_fdata().astype(np.float64)

# Concatenate along the fourth dimension
dwi_synth = np.concatenate((dwi_synth_1[..., np.newaxis], dwi_synth_2[..., np.newaxis]), axis=3)

# Extract specific slices and volumes
B0s = dwi_synth[42:86, 42:86, 28:30, :9]

# Calculate the mean and sum of squares
m_1_data = np.mean(B0s, axis=3)
m_2_data = (1 / 9) * np.sum(B0s ** 2, axis=3)

# Create brain mask
brain_mask = nib.load(r'C:\Users\c1668701\OneDrive - Cardiff University\Documents\GitHub\VIC-HACK-2024_2\NaNN\syn_data/dwi_synth_2.nii.gz').get_fdata().astype(np.float64)
brain_mask = np.ma.any(brain_mask[42:86, 42:86, 28:30, :9],axis=3)

# Find indices of brain mask
ind_brain = np.where(brain_mask)

# Create data for optimization
y_data = np.column_stack((m_1_data[ind_brain], m_2_data[ind_brain]))

# Function to calculate moments of the Rician distribution
def moments_rician(p):
    nu = p.reshape(2, -1).T[:,0]
    sigma = p.reshape(2, -1).T[:,1]

    nu2 = nu ** 2
    sigma2 = sigma ** 2

    x = -nu2 / (2 * sigma2)
    I0_mxp5 = ive(0, -x / 2)
    I1_mxp5 = ive(1, -x / 2)

    L_0p5_x = np.exp(x / 2) * ((1 - x) * I0_mxp5 - x * I1_mxp5)
    mus_rician_1 = sigma * np.sqrt(np.pi / 2) * L_0p5_x
    mus_rician_2 = 2 * sigma2 + nu2

    return np.column_stack((mus_rician_1-m_1_data[ind_brain].flatten(), mus_rician_2-m_1_data[ind_brain].flatten())).flatten()

# Define initial guess
x0 = np.ones((len(ind_brain[0]), 2))
x0[:,0]=x0[:,0]*3
x0[:,1]=x0[:,1]*2.6

sol = least_squares(moments_rician,x0.flatten(),verbose=2)

print(sol)

