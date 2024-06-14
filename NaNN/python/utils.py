import numpy as np
import torch
import copy
import SimpleITK as sitk
import importlib.util
# if importlib.util.find_spec("tensorflow"):
#     import tensorflow as tf

def convert_data(data_np, pack):
    if pack == 'pytorch':
        data = torch.tensor(data_np, dtype=torch.float32)
    return data

def get_loss(name, pack):
    if name == 'rician':
        if pack == 'pytorch':
            from RicianLoss.lossfuncs import ricianloss_ptcl as loss
        elif pack == 'python':
            from RicianLoss.lossfuncs import ricianloss_py as loss
    return loss

def get_model(name, pack):
    if name == 'ivim':
        if pack == 'pytorch':
            from RicianLoss.modelfuncs import ivim_pt as model
    return model

def get_net(name, pack):
    if name == 'qmri':
        if pack == 'pytorch':
            from RicianLoss.nnfuncs import Network_qmri_pt as net
    return net

def get_train(pack):
    if pack == 'pytorch':
        from RicianLoss.nnfuncs import train_network_pt as train
    return train


def get_test(pack):
    if pack == 'pytorch':
        from RicianLoss.nnfuncs import test_network_pt as test
    return test


def get_fitdata(img4d, mask):
    fit_inds = np.where(mask == 1)
    n_vox_fit = len(fit_inds[0])
    n_z = img4d.shape[3]
    img_fitdata = np.ones((n_vox_fit, n_z)) * np.nan
    for i in range(n_vox_fit):
        fit_ind = [fit_inds[0][i], fit_inds[1][i], fit_inds[2][i]]
        img_fitdata[i, :] = img4d[fit_ind[0], fit_ind[1], fit_ind[2], :]
    return img_fitdata

def get_fitdata_3d(img3d, mask):
    fit_inds = np.where(mask == 1)
    n_vox_fit = len(fit_inds[0])
    img_fitdata = np.ones((n_vox_fit)) * np.nan
    for i in range(n_vox_fit):
        fit_ind = [fit_inds[0][i], fit_inds[1][i], fit_inds[2][i]]
        img_fitdata[i] = img3d[fit_ind[0], fit_ind[1], fit_ind[2]]
    return img_fitdata

def get_maps(param_pred, mask):
    n_params = param_pred.shape[1]
    maps4d = np.stack([mask] * n_params,axis=-1) * np.nan
    fit_inds = np.where(mask == 1)
    n_vox = param_pred.shape[0]
    for i in range(n_vox):
        fit_ind = [fit_inds[0][i], fit_inds[1][i], fit_inds[2][i]]
        maps4d[fit_ind[0], fit_ind[1], fit_ind[2], :] = param_pred[i, :]
    return maps4d


def save_imgs3d(param_maps, map_labels, save_base, sitk_meta=None):
    param_maps_sv = copy.deepcopy(param_maps)
    n_params = param_maps.shape[3]
    for m in range(n_params):
        param_map_m_sv = np.transpose(param_maps_sv[:, :, :, m], [2, 0, 1])
        param_map_m_sv = sitk.GetImageFromArray(param_map_m_sv, isVector=False)
        if sitk_meta != None:
            param_map_m_sv.CopyInformation(sitk_meta)
        param_map_m_sv_dir = save_base + '/' + map_labels[m] + '.nii.gz'
        sitk.WriteImage(param_map_m_sv, param_map_m_sv_dir)
    print('Maps saved to ' + save_base)
    return None

def save_img4d(img, name, save_base, sitk_meta=None):
    img_ = copy.deepcopy(img)
    img_tp = np.transpose(img_, [3, 2, 0, 1])
    img_sv = sitk.GetImageFromArray(img_tp, isVector=False)
    if sitk_meta != None:
        img_sv.CopyInformation(sitk_meta)
    #
    sv_dir = save_base + name + '.nii.gz'
    sitk.WriteImage(img_sv, sv_dir) # note: Nans will be read back in as 0 in SITK!
    return None
