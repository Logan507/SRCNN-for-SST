import argparse

import h5py
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import colors

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from utils import normalise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='output/x6/best.pth')
    parser.add_argument('--image-file', type=str, default='new_dataset/test(1601).h5')
    parser.add_argument('--scale', type=int, default=6)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    best_psnr, best_index = 0, 0
    worst_psnr, worst_index = 0, 0
    sum_psnr = 0
    with h5py.File(args.image_file, 'r') as f:

        total = f['lr'].shape[0]
        for index in range(total):
            sst_lr = f['lr'][index]
            sst_hr = f['hr'][index]
            sst_lr_bic = cv.resize(sst_lr, (240, 240), interpolation=cv.INTER_CUBIC)

            vmin = min(np.min(sst_lr_bic), np.min(sst_hr))
            vmax = max(np.max(sst_lr_bic), np.max(sst_hr))
            norm = colors.Normalize(vmin, vmax)

            sst_lr_bic = sst_lr_bic.astype(np.float32)
            max_pixel = max(np.max(sst_lr_bic), np.max(sst_hr))
            sst_lr_norm = normalise(sst_lr_bic, max_pixel)
            sst_hr_norm = normalise(sst_hr, max_pixel)

            y = torch.from_numpy(sst_lr_norm).to(device)
            y = y.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)

            hr = torch.from_numpy(sst_hr_norm).to(device)
            psnr = calc_psnr(preds, hr)
            sum_psnr += psnr
            if psnr > best_psnr:
                best_psnr = psnr
                best_index = index
            print('{} PSNR: {:.2f}'.format(index, psnr))

            preds = preds.mul(max_pixel).cpu().numpy().squeeze(0).squeeze(0)

        print("best psnr:{}".format(best_psnr))
        print("best index:{}".format(best_index))
        print("worst psnr:{}".format(best_psnr))
        print("best index:{}".format(best_index))
        print("average_psnr:{}".format(sum_psnr / total))

    f.close()

    # # draw best sst
    # sst_lr = f['lr'][129]
    # sst_hr = f['hr'][129]
    # sst_lr_bic = cv.resize(sst_lr, (240, 240), interpolation=cv.INTER_CUBIC)
    #
    # vmin = min(np.min(sst_lr_bic), np.min(sst_hr))
    # vmax = max(np.max(sst_lr_bic), np.max(sst_hr))
    # norm = colors.Normalize(vmin, vmax)
    #
    # plt.pcolormesh(sst_lr, norm=norm)
    # plt.colorbar()
    # plt.title('sst_lr[129].jpg')
    # plt.savefig('sst_lr[129].jpg')
    #
    # plt.pcolormesh(sst_lr_bic, norm=norm)
    #
    # plt.title('sst_lr_bic_x6[129].jpg')
    # plt.savefig('sst_lr_bic_x6[129].jpg')
    #
    # sst_lr_bic = sst_lr_bic.astype(np.float32)
    # max_pixel = max(np.max(sst_lr_bic), np.max(sst_hr))
    # sst_lr_norm = normalise(sst_lr_bic, max_pixel)
    # sst_hr_norm = normalise(sst_hr, max_pixel)
    #
    # y = torch.from_numpy(sst_lr_norm).to(device)
    #
    # y = y.unsqueeze(0).unsqueeze(0)
    #
    # with torch.no_grad():
    #     preds = model(y).clamp(0.0, 1.0)
    #
    # hr = torch.from_numpy(sst_hr_norm).to(device)
    # psnr = calc_psnr(preds, hr)
    # print('PSNR: {:.2f}'.format(psnr))
    #
    # preds = preds.mul(max_pixel).cpu().numpy().squeeze(0).squeeze(0)
    #
    # plt.pcolormesh(preds, norm=norm)
    # plt.title('sst_preds_x6[129].jpg')
    # plt.savefig('sst_preds_x6[129].jpg')
    #
    # plt.pcolormesh(sst_hr, norm=norm)
    # plt.title('sst_hr[129].jpg')
    # plt.savefig('sst_hr[129].jpg')
