import h5py
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt

h5_file = r'/\new_dataset\train(18-20).h5'


def normalise(sst, norm_constant):
    sst = sst / norm_constant
    return sst


with h5py.File(h5_file, 'r') as f:
    lr = f['lr'][0]
    hr = f['hr'][0]

    max_pixel = max(np.max(lr), np.max(hr))

    sst_lr_norm = normalise(lr, max_pixel)
    sst_hr_norm = normalise(hr, max_pixel)

    sst_lr_norm = torch.from_numpy(sst_lr_norm)
    sst_lr_norm = torch.unsqueeze(sst_lr_norm, 0)
    sst_lr_norm = sst_lr_norm.type(torch.FloatTensor)

    print("sst_lr:{}".format(sst_lr_norm))
    print("sst_lr shape:{}".format(sst_lr_norm.shape))
    print("sst_lr type:{}".format(type(sst_lr_norm)))

    sst_hr_norm = torch.from_numpy(sst_hr_norm)
    sst_hr_norm = torch.unsqueeze(sst_hr_norm, 0)
    sst_hr_norm = sst_hr_norm.type(torch.FloatTensor)

    print("sst_hr:{}".format(sst_hr_norm))
    print("sst_hr shape:{}".format(sst_hr_norm.shape))
    print("sst_hr type:{}".format(type(sst_hr_norm)))
