import h5py
import numpy as np
import glob
import torch
import cv2 as cv
import matplotlib.pyplot as plt


h5_file = r'new_dataset\test(1601).h5'

with h5py.File(h5_file, 'r') as f:
    lr = f['lr'][12]
    plt.pcolormesh(lr)
    plt.colorbar()
    plt.show()