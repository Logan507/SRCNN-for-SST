import argparse
import glob
import h5py
import numpy as np
import cv2 as cv


def train(args):
    h5_file = h5py.File(args.output_path, 'w')
    h5_file_old = h5py.File(args.images_dir, 'r')

    lr_patches = []
    hr_patches = []

    for i in range(h5_file_old['lr'].shape[0]):
        lr_p = h5_file_old['lr'][i]

        lr_p_up = cv.resize(lr_p, (240, 240), interpolation=cv.INTER_CUBIC)
        hr_p = h5_file_old['hr'][i]

        lr_patches.append(lr_p_up)
        hr_patches.append(hr_p)

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()
    h5_file_old.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')
    h5_file_old = h5py.File(args.images_dir, 'r')

    lr_patches = []
    hr_patches = []

    for i in range(h5_file_old['lr'].shape[0]):
        lr_p = h5_file_old['lr'][i]

        lr_p_up = cv.resize(lr_p, (240, 240), interpolation=cv.INTER_CUBIC)
        hr_p = h5_file_old['hr'][i]

        lr_patches.append(lr_p_up)
        hr_patches.append(hr_p)

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()
    h5_file_old.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default=r'new_dataset/val17.h5')
    parser.add_argument('--output-path', type=str, default=r'val(17).h5')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
