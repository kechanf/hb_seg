from v3dpy.loaders import Raw, PBD
from skimage.transform import rescale
import tifffile
import os
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np

v3d_img_root = "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/Cell_Images"
soma_block_size = (25, 25, 25)  # 50um
# output_img_size = (128, 128, 128) # 128 voxel


meta_info_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/meta_0324.xlsx"
meta_info = pd.read_excel(meta_info_file)

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



def down_sample_v3d_file(source_v3d_img_file, target_full_res_img_file, target_down_sampled_img_file, info_file, down_sampled_soma_file):
    target_full_res_img_nnuet_ver_name = os.path.join(os.path.dirname(target_full_res_img_file), "image_" + os.path.basename(target_full_res_img_file).replace(".tif", "_0000.tif"))
    target_down_sampled_img_nnuet_ver_name = os.path.join(os.path.dirname(target_down_sampled_img_file), "image_" + os.path.basename(target_down_sampled_img_file).replace(".tif", "_0000.tif"))
    target_down_sampled_soma_file = os.path.join(os.path.dirname(down_sampled_soma_file), "image_" + os.path.basename(down_sampled_soma_file).replace(".tif", "_0000.tif"))

    if(os.path.exists(info_file) and os.path.exists(target_full_res_img_nnuet_ver_name) and os.path.exists(target_down_sampled_img_nnuet_ver_name) and os.path.exists(target_down_sampled_soma_file)):
        return
    #
    # if(os.path.exists(info_file) and os.path.exists(target_full_res_img_file) and os.path.exists(target_down_sampled_img_file)):
    #     # rename
    #     # print(f"{target_full_res_img_nnuet_ver_name}")
    #     os.rename(target_full_res_img_file, target_full_res_img_nnuet_ver_name)
    #     os.rename(target_down_sampled_img_file, target_down_sampled_img_nnuet_ver_name)
    #     return


    # pbd = PBD()
    # img = pbd.load(source_v3d_img_file)[0]
    # # img = img.astype("uint8")
    # # tifffile.imwrite(target_full_res_img_nnuet_ver_name, img)
    #
    # origin_size = img.shape
    #
    # img = (rescale(img, 0.5, anti_aliasing=False)*255)
    # img = img.astype(np.float32)
    # img = (img - img.min()) / (img.max() - img.min()) * 255
    # img = img.astype("uint8")
    # tifffile.imwrite(target_down_sampled_img_nnuet_ver_name, img)
    # down_sampled_size = img.shape\
    try:
        img = tifffile.imread(target_down_sampled_img_nnuet_ver_name)
    except:
        pbd = PBD()
        img = pbd.load(source_v3d_img_file)[0]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype("uint8")
        tifffile.imwrite(target_full_res_img_nnuet_ver_name, img)

        origin_size = img.shape
        img = (rescale(img, 0.5, anti_aliasing=False)*255)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype("uint8")
        tifffile.imwrite(target_down_sampled_img_nnuet_ver_name, img)

    try:
        generate_soma_img(img, target_down_sampled_soma_file)
    except Exception as e:
        print(f"Error: {source_v3d_img_file}")
        print(e)

    # to json info
    # info = {
    #     "source_v3d_img_file": source_v3d_img_file,
    #     "target_full_res_img_file": target_full_res_img_file,
    #     "target_down_sampled_img_file": target_down_sampled_img_file,
    #     "origin_size": origin_size,
    #     "down_sampled_size": down_sampled_size
    # }
    # with open(info_file, 'w') as f:
    #     json.dump(info, f)

    print(f"{source_v3d_img_file} done.")

def try_down_sample_v3d_file(source_v3d_img_file, target_full_res_img_file, target_down_sampled_img_file, info_file, down_sampled_soma_file):
    try:
        down_sample_v3d_file(source_v3d_img_file, target_full_res_img_file, target_down_sampled_img_file, info_file, down_sampled_soma_file)
    except Exception as e:
        print(f"Error: {source_v3d_img_file}")
        print(e)

down_sampled_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_img"
info_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/img_info"
full_res_img_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/full_res_img"
down_sampled_soma_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_soma"


def save_nnunet_json(json_file, cell_id, xy_resolution, z_resolution):
    if(os.path.exists(json_file)):
        return
    def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
        with open(file, 'w') as f:
            json.dump(obj, f, sort_keys=sort_keys, indent=indent)
    spacing = (float(z_resolution)/1000, float(xy_resolution) / 1000, float(xy_resolution) / 1000)
    save_json({'spacing': spacing}, json_file)

if __name__ == '__main__':
    v3d_img_files = []
    # walk
    for root, dirs, files in os.walk(v3d_img_root):
        for file in files:
            if file.endswith(".v3dpbd"):
                v3d_img_file = os.path.join(root, file)
                v3d_img_files.append(v3d_img_file)

    os.makedirs(down_sampled_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(full_res_img_dir, exist_ok=True)
    os.makedirs(down_sampled_soma_dir, exist_ok=True)
    # sort
    v3d_img_files.sort()
    # v3d_img_files = v3d_img_files[20000:]

    Parallel(n_jobs=4)(delayed(try_down_sample_v3d_file)(
        source_v3d_img_file,
        os.path.join(full_res_img_dir, os.path.basename(source_v3d_img_file).replace(".v3dpbd", ".tif")),
        os.path.join(down_sampled_dir, os.path.basename(source_v3d_img_file).replace(".v3dpbd", ".tif")),
        os.path.join(info_dir, os.path.basename(source_v3d_img_file).replace(".v3dpbd", ".json")),
        os.path.join(down_sampled_soma_dir, os.path.basename(source_v3d_img_file).replace(".v3dpbd", ".tif"))

    ) for source_v3d_img_file in tqdm(v3d_img_files))

    # meta_info_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/meta_0324.xlsx"
    # meta_info = pd.read_excel(meta_info_file)
    # for line in tqdm(meta_info.iterrows()):
    #     cell_id, xy_resolution, z_resolution = line[1][['cell_id', 'xy_resolution', 'z_resolution']]
    #     # print(cell_id, xy_resolution, z_resolution)
    #     # 补全到5位
    #     try:
    #         cell_id = str(cell_id).zfill(5)
    #         json_file = os.path.join(full_res_img_dir, f"image_{cell_id}.json")
    #         save_nnunet_json(json_file, cell_id, xy_resolution, z_resolution)
    #         json_file = os.path.join(down_sampled_dir, f"image_{cell_id}.json")
    #         save_nnunet_json(json_file, cell_id, xy_resolution, z_resolution)
    #     except:
    #         print(f"Error: {cell_id}")
    #         continue




