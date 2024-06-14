import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the NIfTI files
D_img = nib.load('D.nii.gz')
#D_img = nib.load('dwi_synth_adc.nii.gz')
#adc_img = nib.load('dwi_synth_adc.nii.gz')
adc_img = nib.load('D_MSENN.nii.gz')
adc_img = nib.load('D_RLNN_constsigma.nii.gz')
#adc_img = nib.load('dwi_synth_denoised_adc.nii.gz')
sigma_img = nib.load('sigma.nii.gz')
mask_img = nib.load('mask.nii.gz')

# Get the data arrays from the NIfTI files
D_data = D_img.get_fdata()
adc_data = adc_img.get_fdata()
sigma_data = sigma_img.get_fdata()
mask_data = mask_img.get_fdata()

# Ensure mask is binary (0 and 1)
#mask_data = (mask_data > 0).astype(np.int)
mask_data = (mask_data > 0).astype(np.int32)


# Define the range for D values
min_D = 0  # Set your desired minimum value for D
max_D = 3  # Set your desired maximum value for D

# Compute the voxel-wise difference between D and adc
difference = D_data - adc_data

# Apply the mask to the data
D_values = D_data[mask_data == 1]
difference_values = difference[mask_data == 1]
sigma_values = sigma_data[mask_data == 1]


# Flatten the arrays for scatter plotting
D_values = D_data.flatten()
difference_values = difference.flatten()
sigma_values = sigma_data.flatten()

# Apply the range filter for D values
mask = (D_values >= min_D) & (D_values <= max_D)
D_values_filtered = D_values[mask]
difference_values_filtered = difference_values[mask]
sigma_values_filtered = sigma_values[mask]

# Create a scatter plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(D_values_filtered, difference_values_filtered, c=sigma_values_filtered, cmap='viridis', alpha=0.5, s=1)

# Add color bar
cbar = plt.colorbar(sc)
cbar.set_label('Sigma values')

# Add labels and title
plt.xlabel('D values')
plt.ylabel('Difference (D - adc)')
plt.title(f'Scatter Plot of Voxel-wise  D Differences (Ground-truth - estimated ) vs. Ground-truth D values (Filtered by D range {min_D}-{max_D})\nColor-coded by Sigma values')

# Save the plot as a PNG file
output_file = 'D-adc_scatter_plot.png'
plt.savefig(output_file, format='png', dpi=300)
print(f'Scatter plot saved as {output_file}')

# Show the plot
plt.show()

