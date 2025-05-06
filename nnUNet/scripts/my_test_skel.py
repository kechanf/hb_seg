from skimage.morphology import skeletonize_3d
import tifffile
import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import z_star_ctt
from v3dpy.loaders import Raw, PBD
import os
from scipy.ndimage import zoom
from scipy.ndimage import binary_dilation
from collections import deque
import cc3d
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import rotate
import shutil

#
# # img_file = "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/Cell_Images/13000_13999/13700_13799/13710.v3dpbd"
# # pbd = PBD()
# # img = pbd.load(img_file)[0]
#
# good_sample_ids = ['7964', '13710', '9797', '10160', '13018', '6869', '7720', '7723', '9347', '9359', '9795', '13017', '14359']
# good_sample_ids = [int(f) for f in good_sample_ids]
# # sort
# good_sample_ids.sort()
# z_crop = {}
# z_crop[7723] = (42, -1)
# z_crop[7720] = (45, -1)
# z_crop[9347] = (25, -1)
# z_crop[9359] = (18, -1)
# z_crop[9795] = (30, -1)
# z_crop[9797] = (42, -1)
# z_crop[10160] = (56, -1)
# z_crop[13017] = (28, -1)
# z_crop[13018] = (35, -1)
# z_crop[13710] = (64, -1)
# z_crop[14359] = (34, -1)
#
#
# v3draw_img_root = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"
# mip_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/checked_long_sampels/mip"
# tif_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/checked_long_sampels/tif"
# v3draw_img_list = []
# for root, dirs, files in os.walk(v3draw_img_root):
#     if(not "human_brain_data_v3draw" in root):
#         continue
#     for file in files:
#         if file.endswith('.v3draw'):
#             v3draw_img_list.append(os.path.join(root, file))
#
# for img_file in v3draw_img_list:
#     id = os.path.basename(img_file).split('_')[0]
#     if int(id) not in good_sample_ids:
#         continue
#     raw = Raw()
#     img = raw.load(img_file)[0]
#     img = img.astype("uint8")
#     tif_file = os.path.join(tif_dir, f"{id}.tif")
#     # tifffile.imwrite(tif_file, img)
#
#     if(int(id) in z_crop):
#         img = img[z_crop[int(id)][0]:z_crop[int(id)][1], :, :]
#     mip = np.max(img, axis=0)
#
#     mip_file = os.path.join(mip_dir, f"{id}.png")
#     tifffile.imwrite(mip_file, mip)
#
#
#
#
#

# tif_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/cropped_img_1um"
# mip_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/mip"
#
# tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
# for tif_file in tif_files:
#     img = tifffile.imread(os.path.join(tif_dir, tif_file))
#     mip = np.max(img, axis=0)
#
#     mip_file = os.path.join(mip_dir, tif_file.replace('.tif', '.png'))
#     tifffile.imwrite(mip_file, mip)

# img_file = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit/human_brain_data_v3draw_09664_09811/AtlasVolume/09797_P057_T01_(1)_S007_-_TL.L_R0613_OMZ_20231019_RJ.v3draw"
# raw = Raw()
# img = raw.load(img_file)[0]
# img = img[42:-1, :, :]
# print(img.shape)
# img = img.astype(np.float32)
# img = (img - np.min(img)) / (np.max(img) - np.min(img))
# mip_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/threshold_mips"
#
# for i in range(10):
#     hard_thresholding_t = (0.1 + i * 0.1)
#     img_th = img > hard_thresholding_t
#     img_th = img * img_th
#     mip = np.max(img_th, axis=0)
#
#     mip = mip * 255
#     mip = mip.astype("uint8")
#     mip_file = os.path.join(mip_dir, f"hard_thresholding_{i}.png")
#     tifffile.imwrite(mip_file, mip)

# img_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/3_skel_with_soma/10160_P057_T01_(1)_S004_-_TL.L_R0613_OMZ_20231019_RJ.tif"
# img = tifffile.imread(img_file)
# img = img.astype(np.float32)
# img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
# img  = img.astype("uint8")
# tifffile.imwrite(img_file.replace(".tif", "_skel.tif"), img)

# interested_id = 5782
# img_dirs = ["/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_diff_gamma", "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_equalize", "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_gamma_05",
#             "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_tif", "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_truncated_gamma_05"]
# mip_dir = "/data/kfchen/trace_ws/gamma_trans_test/5782"
# for img_dir in img_dirs:
#     current_img_file = None
#     for img_files in os.listdir(img_dir):
#         id = int(img_files.split('_')[0])
#         if id != interested_id:
#             continue
#         current_img_file = os.path.join(img_dir, img_files)
#         break
#
#     if current_img_file is None:
#         continue
#
#     img = tifffile.imread(current_img_file)
#     mip = np.max(img, axis=0)
#     mip_file = os.path.join(mip_dir, img_dir.split('/')[-1] + ".png")
#     tifffile.imwrite(mip_file, mip)


# tif_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/3_skel_with_soma"
# for tif_file in os.listdir(tif_dir):
#     if(not tif_file.endswith('.tif')):
#         continue
#     img = tifffile.imread(os.path.join(tif_dir, tif_file))
#     print(img.shape)
#     img = img.astype(np.float32)
#     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
#     img = img[:, 600:800, 600:800]
#     img = img.astype("uint8")
#     print(img.shape)
#     tifffile.imwrite(os.path.join(tif_dir, tif_file.replace('.tif', '_croped.tif')), img)

# img_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/3_skel_with_soma/14967.tif"
# img = tifffile.imread(img_file)
# # zoom 下采样
# # img = zoom(img, (0.5, 0.25, 0.25), order=0)
# block_size = (512, 512, 512)
# img_shape = img.shape
# z_start, z_end = max(0, int(img_shape[0] / 2 - block_size[0] / 2)), min(img_shape[0], int(img_shape[0] / 2 + block_size[0] / 2))
# y_start, y_end = max(0, int(img_shape[1] / 2 - block_size[1] / 2)), min(img_shape[1], int(img_shape[1] / 2 + block_size[1] / 2))
# x_start, x_end = max(0, int(img_shape[2] / 2 - block_size[2] / 2)), min(img_shape[2], int(img_shape[2] / 2 + block_size[2] / 2))
# img = img[z_start:z_end, y_start:y_end, x_start:x_end]
# img = binary_dilation(img, iterations=1)
# img = zoom(img, (0.5, 0.5, 0.5), order=0)
# print(img.shape, img.dtype)
# print(np.unique(img), img.max(), img.min())
# img = np.where(img > 0, 100, 0)
# tifffile.imwrite(img_file.replace(".tif", "_temp.tif"), img.astype("uint8"), compression=5)

# img_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/0_seg"
# mip_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/seg_xy_mip"
#
# for img_file in os.listdir(img_dir):
#     if(not img_file.endswith('.tif')):
#         continue
#     img = tifffile.imread(os.path.join(img_dir, img_file))
#     mip = np.max(img, axis=0) * 255
#     mip = mip.astype("uint8")
#     tifffile.imwrite(os.path.join(mip_dir, img_file.replace('.tif', '.png')), mip)
#
# skel_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/1_skel_seg"
# skel_cc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/test_skel_cc"
# meta_info_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/meta_info_0205.xlsx"
# meta_info = pd.read_excel(meta_info_file)
# os.makedirs(skel_cc_dir, exist_ok=True)
#
# def find_cc_3d(img, source_point):
#     source_point = find_true_source_point(img, source_point)
#     if(source_point is None):
#         return None
#     cc = cc3d.connected_components(img)
#     cc_label = cc[source_point[0], source_point[1], source_point[2]]
#     cc_img = np.where(cc == cc_label, 1, 0)
#
#     return cc_img
#
# # 找到最近的前景点
# def find_true_source_point(img, source_point, find_range=25):
#     z_start, z_end = max(0, source_point[0] - find_range), min(img.shape[0], source_point[0] + find_range)
#     y_start, y_end = max(0, source_point[1] - find_range), min(img.shape[1], source_point[1] + find_range)
#     x_start, x_end = max(0, source_point[2] - find_range), min(img.shape[2], source_point[2] + find_range)
#
#     img_block = img[z_start:z_end, y_start:y_end, x_start:x_end]
#     img_block = np.where(img_block > 0, 1, 0)
#     img_block = img_block.astype(np.uint8)
#     if(np.sum(img_block) == 0):
#         return None
#
#     source_point = (source_point[0] - z_start, source_point[1] - y_start, source_point[2] - x_start)
#     # 用np找到最近的前景点
#     foreground_points = np.argwhere(img_block == 1)
#     source_array = np.array(source_point)
#     distances = np.linalg.norm(foreground_points - source_array, axis=1)
#
#     min_index = np.argmin(distances)
#     nearest_point = tuple(foreground_points[min_index])
#
#     nearest_point = (nearest_point[0] + z_start, nearest_point[1] + y_start, nearest_point[2] + x_start)
#     return nearest_point
#
#
# def current_task(skel_file):
#     if (not skel_file.endswith('.tif')):
#         return
#     skel_img = tifffile.imread(os.path.join(skel_dir, skel_file))
#     skel_img = skel_img.astype(np.uint8)
#
#     neuron_id = int(skel_file.split('.')[0])
#     current_meta_info = meta_info[meta_info['cell_id'] == neuron_id]
#     soma_x, soma_y, soma_z = current_meta_info['soma_x'].values[0], current_meta_info['soma_y'].values[0], \
#     current_meta_info['soma_z'].values[0]
#     soma_x, soma_y, soma_z = int(soma_x), int(soma_y), int(soma_z)
#
#     skel_cc_img = find_cc_3d(skel_img, (soma_z, soma_y, soma_x))
#     if (skel_cc_img is None):
#         return
#     tifffile.imwrite(os.path.join(skel_cc_dir, skel_file), skel_cc_img.astype("uint8"), compression=5)
#
#     mip_list = [np.max(skel_img, axis=0), np.max(skel_cc_img, axis=0)]
#     mip_list = [np.where(mip > 0, 255, 0) for mip in mip_list]
#     mip = np.concatenate(mip_list, axis=1)
#     tifffile.imwrite(os.path.join(skel_cc_dir, f"{neuron_id}_mip.png"), mip.astype("uint8"))
#
# # for skel_file in tqdm(os.listdir(skel_dir)):
# #     current_task(skel_file)
#
# Parallel(n_jobs=4)(delayed(current_task)(skel_file) for skel_file in tqdm(os.listdir(skel_dir)))
#
# def get_mapped_somas(neuron_id, ptrs_dir, output_marker_file):
#     if(os.path.exists(output_marker_file)):
#         return pd.read_csv(output_marker_file, sep=',')
#
#     ptrs_files = os.listdir(ptrs_dir)
#     doc_name, xy_resolution, z_resolution, soma_x, soma_y, soma_z = meta_info[meta_info['cell_id'] == neuron_id][
#         ['document_name', 'xy_resolution', 'z_resolution', 'soma_x', 'soma_y', 'soma_z']].values[0]
#     ptrs_file = [f for f in ptrs_files if doc_name in f][0]
#     ptrs_file = os.path.join(ptrs_dir, ptrs_file)
#
#     ptrs_markers = pd.read_csv(ptrs_file, sep=',',
#                                comment='#',
#                                names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
#                                       'color_b'])
#
#     current_soma_pos = ptrs_markers[ptrs_markers['name'] == neuron_id][['x', 'y', 'z']].values[0]
#
#     ptrs_markers['x'] = (ptrs_markers['x'] - current_soma_pos[0] + float(soma_x))
#     ptrs_markers['y'] = (ptrs_markers['y'] - current_soma_pos[1] + float(soma_y))
#     ptrs_markers['z'] = (ptrs_markers['z'] - current_soma_pos[2] + float(soma_z))
#
#     # save
#     ptrs_markers.to_csv(output_marker_file, index=False,  sep=',')
#
#     return ptrs_markers
#
# def get_current_meta_info(neuron_id, meta_info, temp_save_file):
#     if(os.path.exists(temp_save_file)):
#         return pd.read_csv(temp_save_file, sep=',')
#     current_meta_info = meta_info[meta_info['cell_id'] == neuron_id]
#     current_meta_info.to_csv(temp_save_file, index=False, sep=',')
#     return current_meta_info
#
#
# def pad_to_blocksize(image, blocksize):
#     # 检查 blocksize 是否为三维
#     if len(blocksize) != 3:
#         raise ValueError("blocksize must be a tuple of three integers (depth, height, width).")
#
#     # 获取输入图像的尺寸
#     input_shape = image.shape
#
#     # 计算每个维度需要填充的大小
#     pad_depth = max(blocksize[0] - input_shape[0], 0)
#     pad_height = max(blocksize[1] - input_shape[1], 0)
#     pad_width = max(blocksize[2] - input_shape[2], 0)
#
#     # 计算每个维度的前后填充量（保持居中）
#     pad_depth_before = pad_depth // 2
#     pad_depth_after = pad_depth - pad_depth_before
#
#     pad_height_before = pad_height // 2
#     pad_height_after = pad_height - pad_height_before
#
#     pad_width_before = pad_width // 2
#     pad_width_after = pad_width - pad_width_before
#
#     # 使用 np.pad 进行填充
#     padded_image = np.pad(
#         image,
#         pad_width=((pad_depth_before, pad_depth_after),
#                    (pad_height_before, pad_height_after),
#                    (pad_width_before, pad_width_after)),
#         mode='constant',  # 填充值为常数（默认填充0）
#         constant_values=0  # 填充值为0
#     )
#
#     return padded_image
#
# tif_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/image/image_15000_0000.tif"
#
# neuron_id = int(os.path.basename(tif_file).replace('.tif', '').split("_")[1])
# soma_block_root = "/data2/kfchen/tracing_ws/soma_seg/soma_block"
# soma_markers_dir = "/data2/kfchen/tracing_ws/soma_seg/mapped_soma_markers"
# ptrs_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas"
# temp_meta_info_dir = "/data2/kfchen/tracing_ws/soma_seg/temp_meta_info"
# soma_block_dir = os.path.join(soma_block_root, str(neuron_id))
# os.makedirs(soma_block_dir, exist_ok=True)
# soma_block_size = (128, 128, 128) # 50um
# meta_info_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/meta_info_0205.xlsx"
# meta_info = pd.read_excel(meta_info_file)
# output_img_size = (128, 128, 128)
#
#
#
#
# current_meta_info = get_current_meta_info(neuron_id, meta_info, os.path.join(temp_meta_info_dir, f"{neuron_id}.csv"))
# xy_resolution, z_resolution = float(current_meta_info['xy_resolution'].values[0]), float(current_meta_info['z_resolution'].values[0])
# current_block_size = (int(soma_block_size[0] / (z_resolution / 1000)), int(soma_block_size[1] / (xy_resolution / 1000)), int(soma_block_size[2] / (xy_resolution / 1000)))
# img = tifffile.imread(tif_file)
# soma_markers = get_mapped_somas(neuron_id, ptrs_dir, os.path.join(soma_markers_dir, f"{neuron_id}.marker"))
# for soma_marker in tqdm(soma_markers.values):
#     soma_x, soma_y, soma_z = float(soma_marker[0]), float(soma_marker[1]), float(soma_marker[2])
#     z_start, z_end = max(0, int(soma_z - current_block_size[0] / 2)), min(img.shape[0], int(soma_z + current_block_size[0] / 2))
#     y_start, y_end = max(0, int(soma_y - current_block_size[1] / 2)), min(img.shape[1], int(soma_y + current_block_size[1] / 2))
#     x_start, x_end = max(0, int(soma_x - current_block_size[2] / 2)), min(img.shape[2], int(soma_x + current_block_size[2] / 2))
#     if(z_end - z_start != current_block_size[0] or y_end - y_start != current_block_size[1] or x_end - x_start != current_block_size[2]):
#         continue
#
#     soma_block = img[z_start:z_end, y_start:y_end, x_start:x_end]
#     # soma_block = pad_to_blocksize(soma_block, output_img_size)
#     soma_block = (soma_block - np.min(soma_block)) / (np.max(soma_block) - np.min(soma_block)) * 255
#     soma_block = soma_block.astype("uint8")
#     soma_block_file = os.path.join(soma_block_dir, f"{neuron_id}_{int(soma_x)}_{int(soma_y)}_{int(soma_z)}.tif")
#     tifffile.imwrite(soma_block_file, soma_block)
#
#
#     soma_block_mip_file = os.path.join(soma_block_dir, f"{neuron_id}_{int(soma_x)}_{int(soma_y)}_{int(soma_z)}_mip.png")
#
#     #rotate
#     # soma_block = rotate(soma_block, 90, axes=(1, 2), reshape=False)
#
#     mip_list = []
#     for axis in range(3):
#         mip = np.max(soma_block, axis=axis)
#         mip_list.append(mip)
#
#     #z方向转45度
#     temp_soma_block = rotate(soma_block, 45, axes=(0, 2), reshape=False)
#     mip = np.max(temp_soma_block, axis=0)
#     mip_list.append(mip)
#     mip = np.max(temp_soma_block, axis=2)
#     mip_list.append(mip)
#
#     # y方向转45度
#     temp_soma_block = rotate(soma_block, 45, axes=(1, 2), reshape=False)
#     mip = np.max(temp_soma_block, axis=1)
#     mip_list.append(mip)
#     mip = np.max(temp_soma_block, axis=2)
#     mip_list.append(mip)
#
#     #x方向转45度
#     temp_soma_block = rotate(soma_block, 45, axes=(0, 1), reshape=False)
#     mip = np.max(temp_soma_block, axis=0)
#     mip_list.append(mip)
#     mip = np.max(temp_soma_block, axis=1)
#     mip_list.append(mip)
#
#     for i, mip in enumerate(mip_list):
#         mip = (mip - np.min(mip)) / (np.max(mip) - np.min(mip)) * 255
#         mip = mip.astype("uint8")
#         mip_file = os.path.join(soma_block_dir, f"{neuron_id}_{int(soma_x)}_{int(soma_y)}_{int(soma_z)}_mip_{i}.png")
#         tifffile.imwrite(mip_file, mip)
#
#     break

# tif_save_dir = "/data2/kfchen/tracing_ws/soma_seg/tif_image"
# # soma_marker_root = "/PBshare/BRAINTELL/Projects/PTRSB_Terafly"
# soma_marker_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas"
#
# v3dimg_file_list = [
#     "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/PTRSB_DB/P00120-T001-R001-S006-B1/P00120-T001-R001-S006-B1_8bit.v3draw",
#     "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/PTRSB_DB/P00121-T001-R001-S005-B1/P00121-T001-R001-S005-B1_8bit.v3draw",
#     "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/PTRSB_DB/P00122-T001-R001-S006-B1/P00122-T001-R001-S006-B1_8bit.v3draw",
#     "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/PTRSB_DB/P00123-T001-R001-S001-B1/P00123-T001-R001-S001-B1_8bit.v3draw"
# ]
# soma_marker_file_list = [
#     "/PBshare/BRAINTELL/Projects/PTRSB_Terafly/P00120-T001-R001-S006-B1/P00120-T001-R001-S006-B1.apo",
#     "/PBshare/BRAINTELL/Projects/PTRSB_Terafly/P00121-T001-R001-S005-B1/P00121-T001-R001-S005-B1.apo",
#     "/PBshare/BRAINTELL/Projects/PTRSB_Terafly/P00122-T001-R001-S006-B1/P00122-T001-R001-S006-B1.apo",
#     "/PBshare/BRAINTELL/Projects/PTRSB_Terafly/P00123-T001-R001-S001-B1/P00123-T001-R001-S001-B1.apo"
# ]
#
# for v3dimg_file, soma_marker_file in zip(v3dimg_file_list, soma_marker_file_list):
#     # soma_marker_file = "/PBshare/BRAINTELL/Projects/PTRSB_Terafly/P00121-T001-R001-S005-B1/P00121-T001-R001-S005-B1.apo"
#     # v3dimg_file = "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/PTRSB_DB/P00121-T001-R001-S005-B1/P00121-T001-R001-S005-B1_8bit.v3draw"
#     # soma_marker_file = os.path.join(soma_marker_root, os.path.basename(v3dimg_file).replace('_8bit.v3draw', '.apo'))
#
#     shutil.copy(soma_marker_file, os.path.join(soma_marker_dir, os.path.basename(soma_marker_file)))
#     raw = Raw()
#     img = raw.load(v3dimg_file)[0]
#     img = img.astype(np.float32)
#     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
#     img = img.astype("uint8")
#     tifffile.imwrite(os.path.join(tif_save_dir, os.path.basename(v3dimg_file).replace('_8bit.v3draw', '.tif')), img)


img_dir = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_DB/temp_soma_block"
img_files = os.listdir(img_dir)
for img_file in img_files:
    img = tifffile.imread(os.path.join(img_dir, img_file))
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    mip = np.max(img, axis=0)
    mip = mip.astype("uint8")
    tifffile.imwrite(os.path.join(img_dir, img_file.replace('.tif', '_mip.png')), mip)





