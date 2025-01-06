import pandas as pd
import os
import shutil
import tifffile
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
from nnUNet.scripts.mip import get_mip_swc
import cv2
'''
test_img_dir = "/data/kfchen/trace_ws/to_gu/test_img"
test_mip_dir = "/data/kfchen/trace_ws/to_gu/test_img_mip"
for img_file in os.listdir(test_img_dir):
    img = tifffile.imread(os.path.join(test_img_dir,img_file))
    img = np.array(img).astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    mip = np.max(img, axis=0)
    mip = (mip * 255).astype('uint8')
    tifffile.imwrite(os.path.join(test_mip_dir,img_file.replace('.tif', '.png')), mip)
exit()


img_dir = "/data/kfchen/trace_ws/to_gu/img"
test_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
test_img_dir = "/data/kfchen/trace_ws/to_gu/test_img"
mip_dir = "/data/kfchen/trace_ws/to_gu/img_mip"

# test_list = pd.read_csv(test_list_file)["id"].tolist()
img_files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
# for img_file in img_files:
#     if(int(img_file.replace(".tif","")) in test_list):
#         shutil.copy(os.path.join(img_dir,img_file),os.path.join(test_img_dir,img_file))

img_dir = "/data/kfchen/trace_ws/to_gu/mask"
img_files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
mip_dir = "/data/kfchen/trace_ws/to_gu/mask_mip"
def current_task(img_file, mip_file):
    img = tifffile.imread(img_file)
    img = np.array(img).astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    mip = np.max(img, axis=0)
    mip = (mip * 255).astype('uint8')
    tifffile.imwrite(mip_file, mip)

Parallel(n_jobs=10)(delayed(current_task)(
    os.path.join(img_dir,img_file),
    os.path.join(mip_dir,img_file.replace('.tif', '.png'))) for img_file in tqdm(img_files))
'''


def crop_or_pad_image(image, target_size):
    """
    将输入图像剪切或填充到指定大小。

    :param image: 输入的二维图像 (numpy 数组)
    :param target_size: 目标大小 (height, width)
    :return: 处理后的图像
    """
    height, width = image.shape[:2]

    # 目标大小
    target_height, target_width = target_size

    if target_height < height or target_width < width:
        start_y = (height - target_height) // 2
        start_x = (width - target_width) // 2
        if(len(image.shape) == 3):
            cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width, :]
        else:
            cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]
        return cropped_image

        # 如果目标大小大于原图，进行中心填充
    else:
        # 计算需要填充的大小
        pad_top = (target_height - height) // 2
        pad_bottom = target_height - height - pad_top
        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left
        if(len(image.shape) == 3):
            padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        return padded_image

def plot_trace_result(target_id):
    img_dir = "/data/kfchen/trace_ws/to_gu/img"
    trace_result_root = "/data/kfchen/trace_ws/to_gu/classics_recon_result"
    tracers = ['APP2', 'MOST', 'NEUTUBE', 'SNAKE',
               'MST',
               'SimpleAxisAnalyzer', 'NeuronChaser',
               'Advantra']
    trace_result = []
    gs_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/2_flip_after_sort"
    gs_files = [f for f in os.listdir(gs_dir) if f.endswith(".swc")]
    gs_file = [f for f in gs_files if int(f.split(".")[0]) == target_id]
    if (len(gs_file) > 0):
        gs_file = gs_file[0]
        trace_result.append(("Manual", os.path.join(gs_dir, gs_file)))
    for tracer in tracers:
        trace_result_dir = os.path.join(trace_result_root, tracer)
        trace_result_files = [f for f in os.listdir(trace_result_dir) if f.endswith(".swc")]
        trace_result_file = [f for f in trace_result_files if f.split(".")[0] == str(target_id)]

        if (len(trace_result_file) > 0):
            trace_result_file = trace_result_file[0]
            trace_result.append((tracer, os.path.join(trace_result_dir, trace_result_file)))
        else:
            trace_result.append((tracer, None))

    # add proposed?
    # proposed_recon_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/6_connect_soma_swc"
    # proposed_recon_files = [f for f in os.listdir(proposed_recon_dir) if f.endswith(".swc")]
    # proposed_recon_file = [f for f in proposed_recon_files if int(f.split("_")[0]) == int(target_id)]
    # if (len(proposed_recon_file) > 0):
    #     proposed_recon_file = proposed_recon_file[0]
    #     trace_result.append(("proposed", os.path.join(proposed_recon_dir, proposed_recon_file)))

    img = tifffile.imread(os.path.join(img_dir, str(target_id) + ".tif"))
    img = np.array(img).astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    mip = np.max(img, axis=0)

    mip_list = []
    fig_size = (300, 300)
    mip = crop_or_pad_image(mip, fig_size)
    mip = np.flip(mip, axis=0)

    row = 2
    col = 5
    # fig, ax = plt.figure(figsize=(col * 3, row * 3), dpi=300)
    fig, axes = plt.subplots(row, col, figsize=(col * 3, row * 3), dpi=300)
    axes = axes.flatten()
    axes[0].imshow(mip, cmap='gray')
    # plt.imshow(mip, cmap='gray')
    # plt.axis('off')


    for i, (tracer, trace_file) in enumerate(trace_result):
        # plt.subplot(row, col, i + 2)
        ax = axes[i + 1]

        if trace_file is not None:
            current_mip = get_mip_swc(trace_file, img, ignore_background=True)
            # print(f"current_mip.shape: {current_mip.shape}")
            current_mip = crop_or_pad_image(current_mip, fig_size)
            if(tracer == 'proposed' or tracer == 'Manual'):
                current_mip = np.flip(current_mip, axis=0)

            # 添加黑色边框
            current_mip = cv2.copyMakeBorder(current_mip, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

            ax.imshow(current_mip, cmap='gray')
        # ax.title(tracer)
        ax.set_title(tracer, x=0.5, y=0.85, fontsize=15)

    for i in range(row * col):
        ax = axes[i]
        ax.axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"/data/kfchen/trace_ws/to_gu/classics_recon_result/comp/{target_id}.png")


sample_dir = "/data/kfchen/trace_ws/to_gu/sample_test_img"
sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".tif")]
sample_ids = [int(f.split(".")[0]) for f in sample_files]
for target_id in sample_ids:
    plot_trace_result(target_id)








