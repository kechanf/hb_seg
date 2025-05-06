from simple_swc_tool.plot_neuron import plot_img_on_fig, plot_swc_on_fig, plot_markers_on_fig
import os
import tifffile
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.transform import resize



img_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/tif"
swc_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um"
# pruned_swc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/9_my_cut_swc"
# marker_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/mapped_markers"
mip_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/mip"
os.makedirs(mip_dir, exist_ok=True)

meta_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/meta.csv"
meta_info = pd.read_csv(meta_info_file, encoding='gbk')


def current_task(img_file):
    neuron_id = int(img_file.split('_')[0])
    swc_file = os.path.join(swc_dir, img_file.replace('.tif', '.swc'))
    if (not os.path.exists(swc_file)):
        return
    save_file = os.path.join(mip_dir, str(neuron_id) + ".png")
    if (os.path.exists(save_file)):
        return

    xy_resolution = meta_info[meta_info['cell_id'] == neuron_id]['xy_resolution'].values[0]
    z_resolution = 1000
    img = tifffile.imread(os.path.join(img_dir, img_file))
    # 三视图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for axis in range(3):
        img_mip = np.max(img, axis=axis)
        if(axis == 0):
            bkg_shape = (int(img_mip.shape[0] * xy_resolution / 1000), int(img_mip.shape[1] * xy_resolution / 1000))
        elif(axis == 1):
            bkg_shape = (int(img_mip.shape[0] * z_resolution / 1000), int(img_mip.shape[1] * xy_resolution / 1000))
        else:
            bkg_shape = (int(img_mip.shape[0] * z_resolution / 1000), int(img_mip.shape[1] * xy_resolution / 1000))
        img_mip = resize(img_mip, bkg_shape)
        axes[0, axis].imshow(img_mip, cmap='gray')

        background = np.ones(bkg_shape).astype(np.uint8) * 255
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)


        if (axis == 0):
            background = plot_img_on_fig(background, img, projection_direction="xy")
            background = plot_swc_on_fig(background, swc_file, line_color=(255, 0, 0), projection_direction="xy", plot_mode="line")
        elif (axis == 1):
            background = plot_img_on_fig(background, img, projection_direction="xz")
            background = plot_swc_on_fig(background, swc_file, line_color=(255, 0, 0), projection_direction="xz", plot_mode="line")
        else:
            background = plot_img_on_fig(background, img, projection_direction="yz")
            background = plot_swc_on_fig(background, swc_file, line_color=(255, 0, 0), projection_direction="yz", plot_mode="line")

        axes[1, axis].imshow(background)

    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()

def try_current_task(img_file):
    try:
        current_task(img_file)
    except:
        print(f"Error: {img_file}")

img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
# img_files = img_files[:5]
# for img_file in tqdm(img_files):
#     current_task()
Parallel(n_jobs=8)(delayed(try_current_task)(img_file) for img_file in tqdm(img_files))