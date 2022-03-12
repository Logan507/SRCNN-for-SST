import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import normalise


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][idx]
            hr = f['hr'][idx]

            max_pixel = max(np.max(lr), np.max(hr))

            sst_lr_norm = normalise(lr, max_pixel)
            sst_hr_norm = normalise(hr, max_pixel)

            sst_lr_norm = torch.from_numpy(sst_lr_norm)
            sst_lr_norm = torch.unsqueeze(sst_lr_norm, 0)
            sst_lr_norm = sst_lr_norm.type(torch.FloatTensor)

            sst_hr_norm = torch.from_numpy(sst_hr_norm)
            sst_hr_norm = torch.unsqueeze(sst_hr_norm, 0)
            sst_hr_norm = sst_hr_norm.type(torch.FloatTensor)

            return sst_lr_norm, sst_hr_norm

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][idx]
            hr = f['hr'][idx]

            max_pixel = max(np.max(lr), np.max(hr))

            sst_lr_norm = normalise(lr, max_pixel)
            sst_hr_norm = normalise(hr, max_pixel)

            sst_lr_norm = torch.from_numpy(sst_lr_norm)
            sst_lr_norm = torch.unsqueeze(sst_lr_norm, 0)
            sst_lr_norm = sst_lr_norm.type(torch.FloatTensor)

            sst_hr_norm = torch.from_numpy(sst_hr_norm)
            sst_hr_norm = torch.unsqueeze(sst_hr_norm, 0)
            sst_hr_norm = sst_hr_norm.type(torch.FloatTensor)

            return sst_lr_norm, sst_hr_norm

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
