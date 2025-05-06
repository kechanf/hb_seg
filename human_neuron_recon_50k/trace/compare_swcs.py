import os

from numpy.compat import os_PathLike
from tqdm import tqdm
from simple_swc_tool.plot_neuron import plot_img_on_fig, plot_swc_on_fig, plot_markers_on_fig
import tifffile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def plot_swcs(img_file, swc_file1, swc_file2, mip_save_file):
    if(os.path.exists(mip_save_file)):
        return
    if (not os.path.exists(swc_file1) or not os.path.exists(swc_file2) or not os.path.exists(img_file)):
        return

    img = tifffile.imread(os.path.join(img_file))
    img = np.flip(img, axis=1)
    img_mip = np.max(img, axis=0)

    bkg_shape = (int(img_mip.shape[0]), int(img_mip.shape[1]))
    background = np.ones(bkg_shape).astype(np.uint8) * 255
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

    background = plot_img_on_fig(background, img)
    background1, background2 = background.copy(), background.copy()
    background1 = plot_swc_on_fig(background1, swc_file1, line_color=(255, 0, 0), plot_mode="line")
    background2 = plot_swc_on_fig(background2, swc_file2, line_color=(0, 255, 0), plot_mode="line")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(background1)
    ax[1].imshow(background2)

    plt.savefig(mip_save_file)
    plt.close()


if __name__ == "__main__":
    ws_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k"
    img_dir = os.path.join(ws_dir, "down_sampled_img")
    origin_swc_dir = os.path.join(ws_dir, "down_sampled_swcs")
    cut_swc_dir = os.path.join(ws_dir, "down_sampled_retrace_swcs")
    mip_dir = os.path.join(ws_dir, "compare_origin_and_retrace")
    os.makedirs(mip_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    img_files = img_files[:1000]
    Parallel(n_jobs=8)(
        delayed(plot_swcs)(
            os.path.join(img_dir, img_file),
            os.path.join(origin_swc_dir, img_file.replace('_0000.tif', '.swc')),
            os.path.join(cut_swc_dir, img_file.replace('_0000.tif', '.swc')),
            os.path.join(mip_dir, img_file.replace('_0000.tif', '.png'))
        )
        for img_file in tqdm(img_files)
    )