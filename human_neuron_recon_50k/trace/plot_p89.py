import os
import numpy as np
import pandas as pd
import tifffile
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from simple_swc_tool.plot_neuron import plot_img_on_fig
from simple_swc_tool.plot_neuron import plot_swc_on_fig

meta_info_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/meta_0324.xlsx"
meta_info = pd.read_excel(meta_info_file)

ws_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k"
img_dir = os.path.join(ws_dir, "down_sampled_img")
# seg_dir = os.path.join(ws_dir, "down_sampled_seg_with_soma")
# skel_dir = os.path.join(ws_dir, "down_sampled_uint8")
swc_dir = os.path.join(ws_dir, "down_sampled_swcs")
mip_dir = os.path.join(ws_dir, "down_sampled_mips")
os.makedirs(mip_dir, exist_ok=True)

p89_ids = meta_info.loc[meta_info["patient_number"] == "P00089", "cell_id"].values
p89_ids = [int(cell_id) for cell_id in p89_ids]

swc_files = [f for f in os.listdir(swc_dir) if f.endswith(".swc")]
swc_files = [f for f in swc_files if int(f.split("_")[1].split(".")[0]) in p89_ids]
# print(len(swc_files))
img_files = [f.replace(".swc", "_0000.tif") for f in swc_files]
save_files = [f.replace(".swc", ".tif") for f in swc_files]

def plot_p89(img_file, swc_file, save_file):
    if (os.path.exists(save_file)):
        return

    img = tifffile.imread(img_file)
    img = np.flip(img, axis=1)
    img_mip = np.max(img, axis=0)

    bkg_shape = (int(img_mip.shape[0]), int(img_mip.shape[1]))
    background = np.ones(bkg_shape).astype(np.uint8) * 255
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

    background = plot_img_on_fig(background, img)
    background = plot_swc_on_fig(background, swc_file, line_color=(255, 0, 0), plot_mode="line")

    plt.imshow(background)
    plt.savefig(save_file)
    plt.close()

Parallel(n_jobs=8)(
    delayed(plot_p89)(
        os.path.join(img_dir, img_file),
        os.path.join(swc_dir, swc_file),
        os.path.join(mip_dir, save_file))
    for img_file, swc_file, save_file in tqdm(zip(img_files, swc_files, save_files))
)


