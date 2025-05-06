from skimage.morphology import skeletonize_3d
import tifffile
import numpy as np
import pandas as pd
from v3dpy.loaders import Raw, PBD
import os
from scipy.ndimage import zoom
from scipy.ndimage import binary_dilation
from collections import deque
import cc3d
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import rotate
import cv2
import json
from skimage.transform import resize
#
# MODE = "GenerateMIP"
MODE = "GenerateMask"
DATA_TYPE = "PTRSB_DB" # 原始大图像块
# DATA_TYEP = "CELL_BLOCK" # neuron切块

def get_mapped_somas(neuron_id, ptrs_dir, output_marker_file):
    if(os.path.exists(output_marker_file)):
        return pd.read_csv(output_marker_file, sep=',')

    ptrs_files = os.listdir(ptrs_dir)
    doc_name, xy_resolution, z_resolution, soma_x, soma_y, soma_z = meta_info[meta_info['cell_id'] == neuron_id][
        ['document_name', 'xy_resolution', 'z_resolution', 'soma_x', 'soma_y', 'soma_z']].values[0]
    ptrs_file = [f for f in ptrs_files if doc_name in f][0]
    ptrs_file = os.path.join(ptrs_dir, ptrs_file)

    ptrs_markers = pd.read_csv(ptrs_file, sep=',',
                               comment='#',
                               names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
                                      'color_b'])

    current_soma_pos = ptrs_markers[ptrs_markers['name'] == neuron_id][['x', 'y', 'z']].values[0]

    ptrs_markers['x'] = (ptrs_markers['x'] - current_soma_pos[0] + float(soma_x))
    ptrs_markers['y'] = (ptrs_markers['y'] - current_soma_pos[1] + float(soma_y))
    ptrs_markers['z'] = (ptrs_markers['z'] - current_soma_pos[2] + float(soma_z))

    # save
    ptrs_markers.to_csv(output_marker_file, index=False,  sep=',')

    return ptrs_markers

def get_current_meta_info(neuron_id, meta_info, temp_save_file):
    if(os.path.exists(temp_save_file)):
        return pd.read_csv(temp_save_file, sep=',')
    current_meta_info = meta_info[meta_info['cell_id'] == neuron_id]
    current_meta_info.to_csv(temp_save_file, index=False, sep=',')
    return current_meta_info


def pad_to_blocksize(image, blocksize):
    # 检查 blocksize 是否为三维
    if len(blocksize) != 3:
        raise ValueError("blocksize must be a tuple of three integers (depth, height, width).")

    # 获取输入图像的尺寸
    input_shape = image.shape

    # 计算每个维度需要填充的大小
    pad_depth = max(blocksize[0] - input_shape[0], 0)
    pad_height = max(blocksize[1] - input_shape[1], 0)
    pad_width = max(blocksize[2] - input_shape[2], 0)

    # 计算每个维度的前后填充量（保持居中）
    pad_depth_before = pad_depth // 2
    pad_depth_after = pad_depth - pad_depth_before

    pad_height_before = pad_height // 2
    pad_height_after = pad_height - pad_height_before

    pad_width_before = pad_width // 2
    pad_width_after = pad_width - pad_width_before

    # 使用 np.pad 进行填充
    padded_image = np.pad(
        image,
        pad_width=((pad_depth_before, pad_depth_after),
                   (pad_height_before, pad_height_after),
                   (pad_width_before, pad_width_after)),
        mode='constant',  # 填充值为常数（默认填充0）
        constant_values=0  # 填充值为0
    )

    return padded_image


def rotate_img_to_mip(image, rotate_times=12, axes_rot=(1, 2), mip_axis=1):
    rotate_step = int(180 / rotate_times)
    mip_list = []
    for i in range(rotate_times):
        angle = i * rotate_step
        rotated_image = rotate(image, angle, axes=axes_rot, reshape=False)
        mip = np.max(rotated_image, axis=mip_axis)
        # print(mip.shape)
        mip_list.append(mip)

    return mip_list

def generate_2d_mask_from_polygonal(labelme_mask_file):
    if(not os.path.exists(labelme_mask_file)):
        return None
    with open(labelme_mask_file, 'r') as f:
        label_info = json.load(f)
    imageHeight, imageWidth = label_info['imageHeight'], label_info['imageWidth']
    mask = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
    # Polygonal to mask
    # mask_point_list = []
    # print(label_info['shapes'])
    for i in range(len(label_info['shapes'])):
        # mask_point_list.append(np.array(mask['shapes'][i]['points'], dtype=np.int32))
        mask_point = np.array(label_info['shapes'][i]['points'], dtype=np.int32)
        mask = cv2.fillPoly(mask, [mask_point], 1)
    return mask

def generate_3d_mask_from_2d_mip_mask(mip_list, img_shape, rotate_times=12, axes_rot=(1, 2), mip_axis=1):
    total_mask = np.ones((img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8)
    rotate_step = int(180 / rotate_times)
    for i in range(rotate_times):
        mip = mip_list[i]
        if(mip is None):
            continue
        mask = np.expand_dims(mip, axis=mip_axis)
        mask = np.repeat(mask, img_shape[mip_axis], axis=mip_axis)

        angle = i * rotate_step
        mask = rotate(mask, -angle, axes=axes_rot, reshape=False)
        mask = np.where(mask > 0, 1, 0)
        total_mask = total_mask * mask

    return total_mask

def get_mip_file_name(img_file, rotate_times=12, mip_axis=0):
    mip_file_list = []
    rotate_step = int(180 / rotate_times)
    for i in range(rotate_times):
        mip_file_list.append(img_file.replace(".png", f"_axis_{mip_axis}_angle_{i * rotate_step}.tif"))
    return mip_file_list

def limit_pos_in_shape(x, max_x, min_x=0):
    return max(min_x, min(max_x, x))  # 合并 max 和 min 操作

def calculate_bounds(center, block_size, img_shape, axis):
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

def check_file_list_exist(file_list):
    for file in file_list:
        if(not os.path.exists(file)):
            return False
    return True

def find_soma_marker_file_for_ptrs_db(cell_block_ptrs_dir, tif_file):
    ptrs_files = os.listdir(cell_block_ptrs_dir)
    flag = str(os.path.basename(tif_file).replace(".tif", "").replace("_0000", ""))
    for ptrs_file in ptrs_files:
        # if("121" in ptrs_file):
        #     print(ptrs_file)
        #     print(flag)
        if(flag in str(ptrs_file)):
            if(".marker" in ptrs_file or ".apo" in ptrs_file):
                return os.path.join(cell_block_ptrs_dir, ptrs_file)
    return None

soma_block_root = "/data2/kfchen/tracing_ws/soma_seg/soma_block"
soma_markers_dir = "/data2/kfchen/tracing_ws/soma_seg/mapped_soma_markers"
cell_block_ptrs_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas"
temp_meta_info_dir = "/data2/kfchen/tracing_ws/soma_seg/temp_meta_info"
mask_save_dir = "/data2/kfchen/tracing_ws/soma_seg/mask"
meta_info_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/meta_info_0205.xlsx"
meta_info = pd.read_excel(meta_info_file)
soma_block_size = (50, 50, 50)  # 128um
output_img_size = (128, 128, 128) # 128 voxel

if __name__ == '__main__':
    rotate_times = 12
    z_cut = 50
    # tif_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/image/image_15000_0000.tif"
    # tif_file = "/data2/kfchen/tracing_ws/soma_seg/tif_image/P00121-T001-R001-S005-B1_0000.tif"
    tif_file = "/data2/kfchen/tracing_ws/soma_seg/tif_image/P00120-T001-R001-S006-B1_0000.tif"
    # mask_file = "/data2/kfchen/tracing_ws/soma_seg/soma_block/mask_15000_0000.tif"
    mask_file = os.path.join(mask_save_dir, os.path.basename(tif_file))



    if(DATA_TYPE == "CELL_BLOCK"):
        neuron_id = int(os.path.basename(tif_file).replace('.tif', '').split("_")[1])

        soma_block_dir = os.path.join(soma_block_root, str(neuron_id))
        os.makedirs(soma_block_dir, exist_ok=True)

        current_meta_info = get_current_meta_info(neuron_id, meta_info,
                                                  os.path.join(temp_meta_info_dir, f"{neuron_id}.csv"))
        xy_resolution, z_resolution = float(current_meta_info['xy_resolution'].values[0]), float(
            current_meta_info['z_resolution'].values[0])
        current_block_size = (int(soma_block_size[0] / (z_resolution / 1000)),
                              int(soma_block_size[1] / (xy_resolution / 1000)),
                              int(soma_block_size[2] / (xy_resolution / 1000))
                              )

        soma_markers = get_mapped_somas(neuron_id, cell_block_ptrs_dir, os.path.join(soma_markers_dir, f"{neuron_id}.marker"))
    elif(DATA_TYPE == "PTRSB_DB"):
        # ptrs_flag = str(os.path.basename(tif_file).replace('.tif', ''))
        # P00121-T001-R001-S005-B1
        ptrs_flag = os.path.basename(tif_file).replace(".tif", "").replace("_0000", "")

        soma_block_dir = os.path.join(soma_block_root, ptrs_flag)
        os.makedirs(soma_block_dir, exist_ok=True)

        current_meta_info = meta_info[(meta_info['PTRSB'] == ptrs_flag)]

        xy_resolution, z_resolution = 300, 1000
        current_block_size = (int(soma_block_size[0] / (z_resolution / 1000)),
                              int(soma_block_size[1] / (xy_resolution / 1000)),
                              int(soma_block_size[2] / (xy_resolution / 1000))
                              )
        soma_markers_file = find_soma_marker_file_for_ptrs_db(cell_block_ptrs_dir, tif_file)
        if(".apo" in soma_markers_file):
            # n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b
            soma_markers = pd.read_csv(soma_markers_file, sep=',',
                                   comment='#',
                                   names=['n', 'orderinfo', 'name', 'comment', 'z', 'x', 'y', 'pixmax', 'intensity',
                                          'sdev', 'volsize', 'mass', 'comment1', 'comment2', 'comment3', 'color_r', 'color_g', 'color_b'])

        elif(".marker" in soma_markers_file):
            soma_markers = pd.read_csv(soma_markers_file, sep=',',
                                   comment='#',
                                   names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
                                          'color_b'])
        # print(soma_markers)
        # exit()


    img = tifffile.imread(tif_file)
    # img = img[z_cut:, :, :]
    # mip = np.max(img, axis=0)
    # mip = ((mip - np.min(mip)) / (np.max(mip) - np.min(mip)) * 255).astype("uint8")
    # tifffile.imwrite("/data2/kfchen/tracing_ws/soma_seg/soma_block/P00120-T001-R001-S006-B1_0000_mip.tif", mip)
    # exit()

    if(MODE == "GenerateMask"):
        total_mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    for soma_marker in tqdm(soma_markers.values):

        # if(soma_x < 0 or soma_x >= img.shape[2] or soma_y < 0 or soma_y >= img.shape[1] or soma_z < 0 or soma_z >= img.shape[0]):
        #     continue
        if(DATA_TYPE == "PTRSB_DB"):
            soma_x, soma_y, soma_z = float(soma_marker[5]), float(soma_marker[6]), float(soma_marker[4])
            # print(f"soma_x: {soma_x}, soma_y: {soma_y}, soma_z: {soma_z}")
            soma_block_mip_file = os.path.join(soma_block_dir, f"{ptrs_flag}_{int(soma_x)}_{int(soma_y)}_{int(soma_z)}_mip.png")
            current_mask_file = os.path.join(soma_block_dir, f"{ptrs_flag}_{int(soma_x)}_{int(soma_y)}_{int(soma_z)}_mask.tif")
        elif(DATA_TYPE == "CELL_BLOCK"):
            soma_x, soma_y, soma_z = float(soma_marker[0]), float(soma_marker[1]), float(soma_marker[2])
            # print(f"soma_x: {soma_x}, soma_y: {soma_y}, soma_z: {soma_z}")
            soma_block_mip_file = os.path.join(soma_block_dir,
                                               f"{neuron_id}_{int(soma_x)}_{int(soma_y)}_{int(soma_z)}_mip.png")
            current_mask_file = os.path.join(soma_block_dir,
                                                f"{neuron_id}_{int(soma_x)}_{int(soma_y)}_{int(soma_z)}_mask.tif")

        z_start, z_end = calculate_bounds(soma_z, current_block_size[0], img.shape, axis=0)
        y_start, y_end = calculate_bounds(soma_y, current_block_size[1], img.shape, axis=1)
        x_start, x_end = calculate_bounds(soma_x, current_block_size[2], img.shape, axis=2)

        if(z_end <= z_start or y_end <= y_start or x_end <= x_start):
            print(f"current soma out of range: {soma_x}, {soma_y}, {soma_z}, img shape: {img.shape}")
            continue
        # print(f"z_start: {z_start}, z_end: {z_end}, y_start: {y_start}, y_end: {y_end}, x_start: {x_start}, x_end: {x_end}")

        soma_block = img[z_start:z_end, y_start:y_end, x_start:x_end]
        # resize
        rescaled_1um_shape = [soma_block.shape[0] * z_resolution / 1000, soma_block.shape[1] * xy_resolution / 1000,
                                soma_block.shape[2] * xy_resolution / 1000]
        rescaled_1um_shape = [int(i) for i in rescaled_1um_shape]
        soma_block = resize(soma_block, rescaled_1um_shape, order=0, mode='reflect', anti_aliasing=True)

        # if not json exists
        mip_file_list = get_mip_file_name(soma_block_mip_file, rotate_times=rotate_times, mip_axis=0)
        polygonal_label_file_list = [mip_file.replace(".tif", ".json") for mip_file in mip_file_list]
        if(MODE == "GenerateMask"):
            mask_list = []
            for i in range(rotate_times):
                mask = generate_2d_mask_from_polygonal(polygonal_label_file_list[i])
                mask_list.append(mask)
            # print("!!!")
            mask = generate_3d_mask_from_2d_mip_mask(mask_list, soma_block.shape, rotate_times=rotate_times)
            mask = resize(mask, current_block_size, order=0, mode='reflect', anti_aliasing=True)
            # save
            # mask = mask * 255
            # mask = mask.astype("uint8")
            # mask_file = os.path.join(soma_block_dir, f"{neuron_id}_{int(soma_x)}_{int(soma_y)}_{int(soma_z)}_total_mask.tif")
            # tifffile.imwrite(mask_file, mask)

            total_mask[z_start:z_end, y_start:y_end, x_start:x_end] = mask + total_mask[z_start:z_end, y_start:y_end, x_start:x_end]
        else:
            # print("!!!")
            if(check_file_list_exist(mip_file_list)):
                continue
            mip_list = rotate_img_to_mip(soma_block)
            for i in range(rotate_times):
                mip_file = mip_file_list[i]
                mip = mip_list[i]
                mip = (mip - np.min(mip)) / (np.max(mip) - np.min(mip)) * 255
                # print(mip_file)
                tifffile.imwrite(mip_file, mip.astype("uint8"))
    if (MODE == "GenerateMask"):
        total_mask = np.where(total_mask > 0, 255, 0).astype("uint8")
        tifffile.imwrite(mask_file, total_mask, compression=5)

        mask = tifffile.imread(mask_file)
        mip = np.max(mask, axis=0)
        mip = ((mip - np.min(mip)) / (np.max(mip) - np.min(mip)) * 255).astype("uint8")
        # tifffile.imwrite("/data2/kfchen/tracing_ws/soma_seg/soma_block/mip_15000_0000.png", mip.astype("uint8"))
        img = tifffile.imread(tif_file)
        img_mip = np.max(img, axis=0)
        img_mip = ((img_mip - np.min(img_mip)) / (np.max(img_mip) - np.min(img_mip)) * 255).astype("uint8")
        result_mip_dir = "/data2/kfchen/tracing_ws/soma_seg/soma_block/result_mip"
        result_mip_file = os.path.join(result_mip_dir, os.path.basename(tif_file).replace(".tif", ".png"))
        tifffile.imwrite(result_mip_file, np.concatenate([img_mip, mip], axis=1).astype("uint8"))

