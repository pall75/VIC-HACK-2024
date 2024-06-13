# Example qmri network in pytorch with real data
import numpy as np
import pickle
from RicianLoss.utils import convert_data, get_fitdata, get_loss, get_model,\
    get_net, get_train, get_test, get_maps, save_imgs3d
from RicianLoss.sigma import sigma_estimate
from matplotlib import pyplot as plt
# import os
# import inspect
import importlib.util
import pdb
import nibabel

# pdb.set_trace()

# Data
data_fn = importlib.util.find_spec("RicianLoss").submodule_search_locations[0] + '/data/real_data.pkl'
with open(data_fn, "rb") as f:
    dwi, mask, b_values, img_meta_3d, img_meta_4d = pickle.load(f)

f.close()


# Fit array
dwi_fitdata = get_fitdata(dwi, mask)

# Data conversion
train_data_pt = convert_data(dwi_fitdata, 'pytorch')
b_values_pt = convert_data(b_values, 'pytorch') # torch.FloatTensor(b_values)

# qMRI Model
ivim_pt = get_model('ivim','pytorch')

# Sigma (from b0 background)
dwi_b0 = dwi[:,:,:,0]
dwi_b0_bg = dwi_b0[np.where(mask==0)]
sigma_est = sigma_estimate(dwi_b0_bg, 'bg')

sigma_map = nibabel.load('/home/c1668701/Downloads/sigma_map.nii.gz')
sigma_map_fitdata = get_fitdata_3d(sigma_map, mask)

# Loss function
ricianloss_ptcl = get_loss('rician','pytorch')
lossfun = ricianloss_ptcl(sigma_est)

# PyTorch neural network
Network_pt = get_net('qmri','pytorch')
n_lat_params = 4
net = Network_pt(b_values_pt, ivim_pt, n_lat_params)

# Train network
train_network_pt = get_train('pytorch')
max_epochs = 10
patience_n = 3
net_trained = train_network_pt(net, train_data_pt,
        lossfun, max_epochs, patience_n)[0]

# Test network
test_network_pt = get_test('pytorch')
param_pred, test_data_pred = test_network_pt(net_trained, train_data_pt)
param_maps = get_maps(param_pred, mask)


# Plot
plt.scatter(b_values, dwi_fitdata[90000,:], label='Measured')
plt.plot(b_values, test_data_pred[90000,:], label='Predicted')
plt.xlabel('b-value [um2/ms]')
plt.ylabel('Signal')
plt.gca().set_ylim(bottom=0)
plt.legend()
plt.show()


# Save parameter maps
map_labels = ['S0', 'Dp', 'Fp', 'Dt']
save_base = importlib.util.find_spec("RicianLoss").submodule_search_locations[0] + '/data'
# save_base = '/Users/christopherparker/Documents/Projects/deep_qmri/software/JSS_prep/python_package/mypackage/initial_release/RicianLoss/src/RicianLoss/data'
save_imgs3d(param_maps, map_labels, save_base, img_meta_3d)


# Plot parameter map
map_ind = 3
plt.imshow(param_maps[:,:,map_ind])
plt.title(map_labels[map_ind])
plt.colorbar()
plt.show()

