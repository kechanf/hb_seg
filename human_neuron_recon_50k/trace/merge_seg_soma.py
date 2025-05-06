import os
import numpy as np
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from tqdm import tqdm

seg_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_seg"
soma_seg_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_soma_seg"
save_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_seg_with_soma"
log_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/log.txt"

os.makedirs(save_dir, exist_ok=True)

meta_info_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/meta_0324.xlsx"
meta_info = pd.read_excel(meta_info_file)
soma_block_size = (25, 25, 25)  # 50um

def calculate_bounds(center, block_size, img_shape, axis):
    def limit_pos_in_shape(x, max_x, min_x=0):
        return max(min_x, min(max_x, x))  # 合并 max 和 min 操作
    """
    计算起始和结束位置，并限制在图像范围内。
    :param center: 中心点坐标（如 soma_z, soma_y, soma_x）
    :param block_size: 当前块的尺寸
    :param img_shape: 图像的形状
    :param axis: 图像的轴（0, 1, 2 分别对应 z, y, x）
    :return: 限制后的起始和结束位置
    """
    start = int(center - block_size / 2)
    end = int(center + block_size / 2)
    return limit_pos_in_shape(start, img_shape[axis]), limit_pos_in_shape(end, img_shape[axis])

def generate_soma_img(img, save_path):
    # if(os.path.exists(save_path)):
    #     return
    neuron_id = int(os.path.basename(save_path).split("_")[1].split(".")[0])
    # print(neuron_id, meta_info.shape, meta_info[meta_info['cell_id'] == neuron_id])
    xy_resolution, z_resolution, soma_x, soma_y, soma_z = meta_info[meta_info['cell_id'] == neuron_id][
        ['xy_resolution', 'z_resolution', 'soma_x', 'soma_y', 'soma_z']].values[0]
    xy_resolution, z_resolution = float(xy_resolution), float(z_resolution)
    soma_z, soma_y, soma_x = float(soma_z), float(soma_y), float(soma_x)
    soma_z, soma_y, soma_x = soma_z /2, soma_y /2, soma_x /2
    # print(soma_z, soma_y, soma_x)

    # xy_resolution, z_resolution = float(current_meta_info['xy_resolution'].values[0]), float(
    #     current_meta_info['z_resolution'].values[0])
    current_block_size = (int(soma_block_size[0] / (z_resolution / 1000)),
                          int(soma_block_size[1] / (xy_resolution / 1000)),
                          int(soma_block_size[2] / (xy_resolution / 1000))
                          )
    z_start, z_end = calculate_bounds(soma_z, current_block_size[0], img.shape, axis=0)
    y_start, y_end = calculate_bounds(soma_y, current_block_size[1], img.shape, axis=1)
    x_start, x_end = calculate_bounds(soma_x, current_block_size[2], img.shape, axis=2)

    # img = np.flip(img, axis=1)
    soma_block = img[z_start:z_end, y_start:y_end, x_start:x_end]
    tifffile.imwrite(save_path, soma_block)

def expand_soma_to_origin_size(seg_file, soma_seg_file):
    seg = tifffile.imread(seg_file)
    soma = tifffile.imread(soma_seg_file)
    seg, soma = seg.astype(np.uint8), soma.astype(np.uint8)
    seg, soma = np.where(seg > 0, 1, 0), np.where(soma > 0, 1, 0)

    neuron_id = int(os.path.basename(seg_file).split("_")[1].split(".")[0])
    xy_resolution, z_resolution, soma_x, soma_y, soma_z = meta_info[meta_info['cell_id'] == neuron_id][
        ['xy_resolution', 'z_resolution', 'soma_x', 'soma_y', 'soma_z']].values[0]
    # print(xy_resolution, z_resolution)
    xy_resolution, z_resolution = float(xy_resolution), float(z_resolution)
    soma_z, soma_y, soma_x = float(soma_z), float(soma_y), float(soma_x)
    soma_z, soma_y, soma_x = soma_z / 2, soma_y / 2, soma_x / 2

    current_block_size = (int(soma_block_size[0] / (z_resolution / 1000)),
                          int(soma_block_size[1] / (xy_resolution / 1000)),
                          int(soma_block_size[2] / (xy_resolution / 1000))
                          )
    z_start, z_end = calculate_bounds(soma_z, current_block_size[0], seg.shape, axis=0)
    y_start, y_end = calculate_bounds(soma_y, current_block_size[1], seg.shape, axis=1)
    x_start, x_end = calculate_bounds(soma_x, current_block_size[2], seg.shape, axis=2)

    bkg = np.zeros_like(seg)
    bkg[z_start:z_end, y_start:y_end, x_start:x_end] = bkg[z_start:z_end, y_start:y_end, x_start:x_end] + soma
    bkg = np.where(bkg > 0, 1, 0)
    return bkg

def merge_seg_and_soma_seg(seg_file, soma_seg_file, save_file):
    if(os.path.exists(save_file)):
        return
    if(not os.path.exists(seg_file) or not os.path.exists(soma_seg_file)):
        return

    seg = tifffile.imread(seg_file)
    soma = tifffile.imread(soma_seg_file)
    seg, soma = seg.astype(np.uint8), soma.astype(np.uint8)
    seg, soma = np.where(seg > 0, 1, 0), np.where(soma > 0, 1, 0)
    if(soma.sum() <= soma.shape[0] * soma.shape[1] * soma.shape[2] / 100):
        # 输出到日志
        with open(log_file, "a") as f:
            f.write(f"Warning: soma size is too small: {soma.sum()} for file {soma_seg_file}\n")
        # print(f"Warning: soma size is too small: {soma.sum()}")
        return

    neuron_id = int(os.path.basename(save_file).split("_")[1].split(".")[0])
    # print(neuron_id, meta_info.shape, meta_info[meta_info['cell_id'] == neuron_id])
    xy_resolution, z_resolution, soma_x, soma_y, soma_z = meta_info[meta_info['cell_id'] == neuron_id][
        ['xy_resolution', 'z_resolution', 'soma_x', 'soma_y', 'soma_z']].values[0]
    xy_resolution, z_resolution = float(xy_resolution), float(z_resolution)
    soma_z, soma_y, soma_x = float(soma_z), float(soma_y), float(soma_x)
    soma_z, soma_y, soma_x = soma_z / 2, soma_y / 2, soma_x / 2
    # print(soma_z, soma_y, soma_x)

    # xy_resolution, z_resolution = float(current_meta_info['xy_resolution'].values[0]), float(
    #     current_meta_info['z_resolution'].values[0])
    current_block_size = (int(soma_block_size[0] / (z_resolution / 1000)),
                          int(soma_block_size[1] / (xy_resolution / 1000)),
                          int(soma_block_size[2] / (xy_resolution / 1000))
                          )
    z_start, z_end = calculate_bounds(soma_z, current_block_size[0], seg.shape, axis=0)
    y_start, y_end = calculate_bounds(soma_y, current_block_size[1], seg.shape, axis=1)
    x_start, x_end = calculate_bounds(soma_x, current_block_size[2], seg.shape, axis=2)

    # img = np.flip(img, axis=1)
    seg[z_start:z_end, y_start:y_end, x_start:x_end] = seg[z_start:z_end, y_start:y_end, x_start:x_end] + soma
    seg = np.where(seg > 0, 1, 0)
    tifffile.imwrite(save_file, seg, compression=5)

    # tifffile.imwrite(save_file.replace(".tif", "_soma.tif"), soma, compression=5)

def try_merge_seg_and_soma_seg(seg_file, soma_seg_file, save_file):
    try:
        merge_seg_and_soma_seg(seg_file, soma_seg_file, save_file)
    except Exception as e:
        # 输出到日志
        with open(log_file, "a") as f:
            f.write(f"Error: {e} for file {seg_file}\n")
        print(f"Error: {e} for file {seg_file}")

if __name__ == "__main__":

    files = [f for f in os.listdir(seg_dir) if f.endswith(".tif")]
    # files = files[20000:20010]
    # for file in files:
    #     seg_file = os.path.join(seg_dir, file)
    #     soma_seg_file = os.path.join(soma_seg_dir, file)
    #     save_file = os.path.join(save_dir, file)
    #     merge_seg_and_soma_seg(seg_file, soma_seg_file, save_file)
    Parallel(n_jobs=4)(
        delayed(try_merge_seg_and_soma_seg)(
            os.path.join(seg_dir, file),
            os.path.join(soma_seg_dir, file),
            os.path.join(save_dir, file)
        ) for file in tqdm(files)
    )

    # bad case
    # 14931 28473 43985 63144 55046
