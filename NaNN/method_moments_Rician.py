import nibabel as nib
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from scipy.optimize import curve_fit
from scipy.special import ive  # Modified Bessel function of the first kind

# Function to calculate moments of the Rician distribution
def moments_rician(xdata, sigma, nu):
    nu2 = nu ** 2
    sigma2 = sigma ** 2

    x = -nu2 / (2 * sigma2)
    I0_mxp5 = ive(0, -x / 2)
    I1_mxp5 = ive(1, -x / 2)
    
    L_0p5_x = np.exp(x / 2) * ((1 - x) * I0_mxp5 - x * I1_mxp5)
    mus_rician_1 = sigma * np.sqrt(np.pi / 2) * L_0p5_x
    mus_rician_2 = 2 * sigma2 + nu2
    
    return np.vstack((mus_rician_1, mus_rician_2)).T

# Read NIfTI files and convert to double
dwi_synth_1 = nib.load('/home/c1668701/VIC-HACK-2024/NaNN/syn_data/dwi_synth_1.nii.gz').get_fdata().astype(np.float64)
dwi_synth_2 = nib.load('/home/c1668701/VIC-HACK-2024/NaNN/syn_data/dwi_synth_2.nii.gz').get_fdata().astype(np.float64)

# Concatenate along the fourth dimension
dwi_synth = np.concatenate((dwi_synth_1[..., np.newaxis], dwi_synth_2[..., np.newaxis]), axis=3)

# Extract specific slices and volumes
B0s = dwi_synth[42:86, 42:86, 28:30, :9]

# Calculate the mean and sum of squares
m_1_data = np.mean(B0s, axis=3)
m_2_data = (1 / 9) * np.sum(B0s ** 2, axis=3)

# Create brain mask
brain_mask = nib.load('/home/c1668701/VIC-HACK-2024/NaNN/syn_data/dwi_synth_2.nii.gz').get_fdata().astype(np.float64)
brain_mask = brain_mask[42:86, 42:86, 28:30, :9]

# Find indices of brain mask
ind_brain = np.where(brain_mask)

# Create data for optimization
y_data = np.column_stack((m_1_data[ind_brain], m_2_data[ind_brain]))

# Define initial guess
x0 = np.ones((len(ind_brain[0]), 2))

# Flatten y_data for curve_fit
y_data_flat = y_data.ravel()

# Define a function for curve_fit
def fit_func(xdata_flat, sigma, nu):
    xdata = xdata_flat.reshape(-1, 2)
    return moments_rician(xdata, sigma, nu).ravel()

# Perform curve fitting
popt, pcov = curve_fit(fit_func, x0.ravel(), y_data_flat, p0=[1, 1])

# Reshape the optimized parameters back to original form
pars_hat = popt.reshape(-1, 2)

print(pars_hat)

