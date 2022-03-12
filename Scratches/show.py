import h5py
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt
from matplotlib import colors

h5_file = r'/\new_dataset\train(18-20).h5'

with h5py.File(h5_file, 'r') as f:
    lr = f['lr'][99]
    hr = f['hr'][99]

    vmin = min(np.min(lr), np.min(hr))
    vmax = max(np.max(lr), np.max(hr))
    norm = colors.Normalize(vmin, vmax)

    plt.pcolormesh(hr, norm=norm)
    plt.colorbar()
    plt.title('sst_hr[99]')
    plt.show()

