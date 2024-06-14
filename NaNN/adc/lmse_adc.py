import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# Load the DWI data
dwi='dwi_synth_denoised'
dwi_nii=dwi + '.nii.gz'
dwi_bvals =dwi + '.bval'
dwi_bvecs =dwi + '.bvec'
dwi_img = nib.load(dwi_nii)
dwi_data = dwi_img.get_fdata()

# Load b-values and b-vectors
bvals, bvecs = read_bvals_bvecs(dwi_bvals, dwi_bvecs)
gtab = gradient_table(bvals, bvecs)

# Prepare the design matrix (b-values)
bvals = np.asarray(bvals)
X = -bvals.reshape(-1, 1)

def fit_adc_for_voxel(i, j, k):
    S = dwi_data[i, j, k, :]
    mask = S > 0
    S = S[mask]
    X_masked = X[mask]

    if len(S) > 1:
        log_S = np.log(S)
        reg = LinearRegression().fit(X_masked, log_S)
        return reg.coef_[0]
    else:
        return 0

# Perform the least-squares fit for each voxel in parallel
adc_map = Parallel(n_jobs=-1)(delayed(fit_adc_for_voxel)(i, j, k)
                              for i in range(dwi_data.shape[0])
                              for j in range(dwi_data.shape[1])
                              for k in range(dwi_data.shape[2]))

# Reshape the result to the original data shape
adc_map = np.array(adc_map).reshape(dwi_data.shape[:3])
adc_map =  adc_map*1000 # Matching the units of the D.nii.gz from synth

# Save the ADC map as a new NIfTI file
adc_img = nib.Nifti1Image(adc_map, dwi_img.affine)
adc_out= dwi + '_adc.nii.gz'
nib.save(adc_img, adc_out)

print("ADC fitting using least-squares approach with parallel processing completed successfully!")

