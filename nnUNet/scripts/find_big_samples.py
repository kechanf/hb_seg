import os.path
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import tifffile
from tqdm import tqdm
from pylib.file_io import load_image
from skimage.transform import resize
import shutil
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from nnUNet.scripts.Elimination_of_fluorescence import find_best_sigma_map, deflu_gamma
from batchgenerators.utilities.file_and_folder_operations import *
from nnUNet.scripts.mip import get_mip_swc

full_meta_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
recon_meta_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_meta.csv"
recon_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_list.csv"
long_3341_to_be_add_root = "/data2/kfchen/tracing_ws/14k_raw_img_data/long_590_to_be_add"
long_3341_to_be_add_img_dir = os.path.join(long_3341_to_be_add_root, "img")
long_3341_to_be_add_mip_dir = os.path.join(long_3341_to_be_add_root, "mip")
#
# if (False):
#     # os.remove(recon_meta_file)
#     if(not os.path.exists(recon_meta_file)):
#         full_meta = pd.read_excel(full_meta_file)
#         recon_list = pd.read_csv(recon_list_file)['id'].to_list()
#         recon_meta = full_meta[full_meta['cell_id'].isin(recon_list)]
#
#         recon_meta.to_csv(recon_meta_file, index=False)
#     else:
#         recon_meta = pd.read_csv(recon_meta_file)
#     print(recon_meta.shape)
#
#     tif_img_dir = "/data/kfchen/trace_ws/de_flu_test/14k_tif"
#     tif_img_list = [f for f in os.listdir(tif_img_dir) if f.endswith('.tif')]
#     img_file_id_map = {}
#     recon_meta['x_shape_1um'] = 0
#     recon_meta['y_shape_1um'] = 0
#     recon_meta['z_shape_1um'] = 0
#     for img_file in tif_img_list:
#         img_file_id_map[int(img_file.split('_')[0])] = img_file
# if(False): # 补充shape信息
#     todo_num = 0
#     def process_row(i, recon_meta, tif_img_dir, img_file_id_map):
#         """
#         处理单行数据，计算 x_shape_1um, y_shape_1um, z_shape_1um。
#
#         参数:
#             i (int): 行索引
#             recon_meta (pd.DataFrame): 数据集
#             tif_img_dir (str): 图像文件目录
#             img_file_id_map (dict): cell_id 到图像文件名的映射
#
#         返回:
#             tuple: (行索引, x_shape_1um, y_shape_1um, z_shape_1um)
#         """
#         try:
#             cell_id = recon_meta.at[i, 'cell_id']
#             img_file = os.path.join(tif_img_dir, img_file_id_map[cell_id])
#             img = tifffile.imread(img_file)
#             img_shape = img.shape
#             xy_resolution = recon_meta.at[i, 'xy_resolution']
#
#             x_shape_1um = img_shape[2] / 1000.0 * xy_resolution
#             y_shape_1um = img_shape[1] / 1000.0 * xy_resolution
#             z_shape_1um = img_shape[0]  # 假设 z 方向不需要乘以分辨率
#
#             return (i, x_shape_1um, y_shape_1um, z_shape_1um)
#         except Exception as e:
#             print(f"Error processing row {i}: {e}")
#             return (i, None, None, None)
#
#
#     # joblib
#     results = Parallel(n_jobs=20)(
#         delayed(process_row)(i, recon_meta, tif_img_dir, img_file_id_map)
#         for i in tqdm(range(len(recon_meta)), desc="Processing rows")
#     )
#
#     for res in results:
#         i, x, y, z = res
#         if x is not None and y is not None and z is not None:
#             recon_meta.at[i, 'x_shape_1um'] = x
#             recon_meta.at[i, 'y_shape_1um'] = y
#             recon_meta.at[i, 'z_shape_1um'] = z
#         else:
#             print(f"Skipping row {i} due to processing error.")
#
#     recon_meta.to_csv(recon_meta_file, index=False)
#
# if(False): # 观察较大图像样例
#     # sort by soma_x
#     recon_meta = recon_meta.sort_values(by=['x_shape_1um'])
#
#     mip_dir = "/data/kfchen/trace_ws/de_flu_test/big_samples_mip"
#     sample_num = 100
#     if(not os.path.exists(mip_dir)):
#         os.makedirs(mip_dir)
#         for i in tqdm(range(sample_num)):
#             id = recon_meta.iloc[i]['cell_id']
#             img_file = os.path.join(tif_img_dir, img_file_id_map[id])
#             img = tifffile.imread(img_file)
#             mip = np.max(img, axis=0)
#             mip_file = os.path.join(mip_dir, str(id) + '_mip.tif')
#             tifffile.imwrite(mip_file, mip)
#
#
# v3draw_img_root = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"
# # walk
# v3draw_img_list, id_list_14k = [], []
# for root, dirs, files in os.walk(v3draw_img_root):
#     if(not "human_brain_data_v3draw" in root):
#         continue
#     for file in files:
#         if file.endswith('.v3draw'):
#             v3draw_img_list.append(os.path.join(root, file))
#             id_list_14k.append(int(file.split('_')[0]))
#
# target_img_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/tif"
# target_mip_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/mip"
# rescaled_1um_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_1um_tif"
# rescaled_500nm_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/rescaled_500nm_tif"
# os.makedirs(target_img_dir, exist_ok=True)
# os.makedirs(target_mip_dir, exist_ok=True)
# os.makedirs(rescaled_1um_dir, exist_ok=True)
# os.makedirs(rescaled_500nm_dir, exist_ok=True)
# full_meta = pd.read_excel(full_meta_file)
# full_meta = full_meta[full_meta['cell_id'].isin(id_list_14k)]
# print("total cell num: ", len(full_meta))
# # v3draw_img_list = v3draw_img_list[:3]
#
# def process_v3draw_img(v3draw_img, target_img_file, target_mip_file, rescaled_1um_file, rescaled_500nm_file):
#     # if(os.path.exists(target_mip_file) and os.path.exists(rescaled_1um_file) and os.path.exists(rescaled_500nm_file)):
#     #     return
#     if(os.path.exists(target_mip_file)):
#         return
#     try:
#         img = load_image(v3draw_img)[0]
#         img = np.array(img).astype(np.float32)
#         img = (img - np.min(img)) / (np.max(img) - np.min(img))
#         tifffile.imwrite(target_img_file, (img*255).astype("uint8"))
#
#         # mip = np.max(img, axis=0)
#         # mip = ((mip - np.min(mip)) / (np.max(mip) - np.min(mip)) * 255).astype("uint8")
#         # tifffile.imwrite(target_mip_file, mip)
#         mips = [np.max(img, axis=0), np.max(img, axis=1), np.max(img, axis=2)]
#         mips = [((mip - np.min(mip)) / (np.max(mip) - np.min(mip)) * 255).astype("uint8") for mip in mips]
#         fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#         ax[0].imshow(mips[0], cmap='gray')
#         ax[1].imshow(mips[1], cmap='gray')
#         ax[2].imshow(mips[2], cmap='gray')
#         for a in ax:
#             a.axis('off')
#         plt.tight_layout()
#         plt.savefig(target_mip_file)
#         plt.close()
#
#         # xy_resolution = full_meta[full_meta['cell_id'] == int(os.path.basename(v3draw_img).split('_')[0])]['xy_resolution'].values[0]
#         # img_shape = img.shape
#         #
#         # img_1um = resize(img, (img_shape[0], int(img_shape[1] / 1000.0 * xy_resolution),
#         #                        int(img_shape[2] / 1000.0 * xy_resolution)), order=1, preserve_range=True)
#         # img_1um = ((img_1um - np.min(img_1um)) / (np.max(img_1um) - np.min(img_1um)) * 255).astype("uint8")
#         # tifffile.imwrite(rescaled_1um_file, img_1um)
#         #
#         # img_500nm = resize(img, (img_shape[0], int(img_shape[1] / 500.0 * xy_resolution),
#         #                         int(img_shape[2] / 500.0 * xy_resolution)), order=1, preserve_range=True)
#         # img_500nm = ((img_500nm - np.min(img_500nm)) / (np.max(img_500nm) - np.min(img_500nm)) * 255).astype("uint8")
#         # tifffile.imwrite(rescaled_500nm_file, img_500nm)
#     except Exception as e:
#         print(f"Error processing {v3draw_img}: {e}")
#
# Parallel(n_jobs=8)(
#     delayed(process_v3draw_img)(v3draw_img_list[i],
#                                 os.path.join(target_img_dir, os.path.basename(v3draw_img_list[i]).replace('.v3draw', '.tif')),
#                                 os.path.join(target_mip_dir, os.path.basename(v3draw_img_list[i]).replace('.v3draw', '.png')),
#                                 os.path.join(rescaled_1um_dir, os.path.basename(v3draw_img_list[i]).replace('.v3draw', '.tif')),
#                                 os.path.join(rescaled_500nm_dir, os.path.basename(v3draw_img_list[i]).replace('.v3draw', '.tif'))
#                                 )
#     for i in tqdm(range(len(v3draw_img_list)), desc="Processing v3draw images")
# )
# # for i in tqdm(range(len(v3draw_img_list)), desc="Processing v3draw images"):
# #     process_v3draw_img(v3draw_img_list[i],
# #                        os.path.join(target_img_dir, os.path.basename(v3draw_img_list[i]).replace('.v3draw', '.tif')),
# #                        os.path.join(target_mip_dir, os.path.basename(v3draw_img_list[i]).replace('.v3draw', '.png')),
# #                        os.path.join(rescaled_1um_dir, os.path.basename(v3draw_img_list[i]).replace('.v3draw', '.tif')),
# #                        os.path.join(rescaled_500nm_dir, os.path.basename(v3draw_img_list[i]).replace('.v3draw', '.tif'))
# #                        )
#
# id_img_map = {}
# for i in range(len(v3draw_img_list)):
#     id = int(os.path.basename(v3draw_img_list[i]).split('_')[0])
#     id_img_map[id] = v3draw_img_list[i]
#
# # del id
# # step 1: del recon id
# recon_list = pd.read_csv(recon_list_file)['id'].to_list()
# for id in recon_list:
#     if id in id_img_map:
#         del id_img_map[id]
#
# # step 2: del mutineuron id
# mutineuron_id_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/mutineuron_list.csv"
# mutineuron_id_list = pd.read_csv(mutineuron_id_list_file)['id'].to_list()
# for id in mutineuron_id_list:
#     if id in id_img_map:
#         del id_img_map[id]
# # print(len(id_img_map))
#
# # step 3: del nk hospital id
# full_meta = pd.read_excel(full_meta_file)
# full_meta = full_meta[full_meta['cell_id'].isin(id_img_map.keys())]
# patient_info_file = "/data/kfchen/trace_ws/patient_info.xlsx"
# patient_info_df = pd.read_excel(patient_info_file)
# # print(full_meta.columns)
# # print(full_meta['patient_number'])
# # print(patient_info_df['patient_number'])
# full_meta['sample_id'] = full_meta['patient_number'].apply(
#     lambda x: patient_info_df.loc[patient_info_df['patient_number'] == ("P" + str(int(x[1:])).zfill(5)), 'sample_id'].values[0]
#     if len(patient_info_df.loc[patient_info_df['patient_number'] == ("P" + str(int(x[1:])).zfill(5)), 'sample_id'].values) > 0
#     else None)
# full_meta['hospital'] = full_meta['sample_id'].apply(lambda x: str(x).split("-")[1] if x is not None else None)
# nk_hospital_list = full_meta[full_meta['hospital'] == 'NK']['cell_id'].to_list()
# for id in nk_hospital_list:
#     if id in id_img_map:
#         del id_img_map[id]
#
# # print(len(id_img_map))
# # 3341 remaining
#
# # step 4: del non-tiling id
# full_meta = pd.read_excel(full_meta_file)
# full_meta = full_meta[full_meta['cell_id'].isin(id_img_map.keys())]
# non_tiling_list = full_meta[full_meta['tiling'] == "0"]['cell_id'].to_list()
# for id in non_tiling_list:
#     if id in id_img_map:
#         del id_img_map[id]
#
# print(len(id_img_map))
# # 590 remaining
#
# # new to add files
#
#
# if(not os.path.exists(long_3341_to_be_add_img_dir)):
#     os.makedirs(long_3341_to_be_add_img_dir)
#     os.makedirs(long_3341_to_be_add_mip_dir)
#
#     for id in tqdm(id_img_map.keys()):
#         file_name = os.path.basename(id_img_map[id])
#         print(file_name)
#         shutil.copyfile(os.path.join(target_img_dir, file_name.replace('.v3draw', '.tif')),
#                         os.path.join(long_3341_to_be_add_img_dir, file_name.replace('.v3draw', '.tif')))
#         shutil.copyfile(os.path.join(target_mip_dir, file_name.replace('.v3draw', '.png')),
#                         os.path.join(long_3341_to_be_add_mip_dir, file_name.replace('.v3draw', '.png')))
#
#
# full_meta = pd.read_excel(full_meta_file)
# full_meta = full_meta[full_meta['cell_id'].isin(id_img_map.keys())]
# def generate_test_data(test_source, imagests, origin_new_id_map_file):
#     def get_xy_resolution(img_path):
#         xy_resolution = full_meta[full_meta['cell_id'] == int(os.path.basename(img_path).split('_')[0])]['xy_resolution'].values[0]
#         return xy_resolution
#
#     def my_augment_gamma(img, xy_resolution):
#         origin_img = img.copy()
#         origin_img_shape = img.shape
#
#         if(xy_resolution == None):
#             return img_file, None
#         img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
#         resolution = (1, xy_resolution/1000, xy_resolution/1000)
#         img = resize(img, (img.shape[0] * resolution[0], img.shape[1] * resolution[1], img.shape[2] * resolution[2]),
#                      order=1)
#         # img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
#
#         soma = np.where(img > 0.9, img, 0).astype(np.float32)
#
#         best_sigma_map, best_sigma = find_best_sigma_map(img, soma)
#         best_sigma_map = resize(best_sigma_map, origin_img_shape, order=3, preserve_range=True, anti_aliasing=False).astype(best_sigma_map.dtype)
#         # tifffile.imwrite(result_img_file.replace('.tif', '_gamma_map.tif'), best_sigma_map)
#         result_img = deflu_gamma(origin_img, best_sigma_map)
#
#         result_img = ((result_img - result_img.min()) / (result_img.max() - result_img.min()) * 255).astype("uint8")
#         return result_img
#
#     data = {
#         'ID': [],
#         'full_name': [],
#         'nnunet_name': [],
#         'spacing': [],
#         'img_size': [],
#         'v3draw_path': []
#     }
#
#     images = [f for f in os.listdir(test_source) if f.endswith('.tif')]
#     images = [os.path.join(test_source, f) for f in images]
#     ids = [int(os.path.basename(im).split('_')[0]) for im in images]
#
#     progress_bar = tqdm(total=len(images), desc="Copying img", unit="file")
#     for im, id in zip(images, ids):
#         progress_bar.update(1)
#         if("IHC" in im):
#             continue
#         target_name = f'image_{(id):03d}'
#
#         img_size = [1, 1, 1]
#         xy_resolution = get_xy_resolution(im)
#         spacing = (1, float(xy_resolution/1000.0), float(xy_resolution/1000.0))
#
#         try:
#             img = tifffile.imread(im)
#             img_size = img.shape
#             if (not os.path.exists(os.path.join(imagests, target_name + '_0000.tif'))):
#                 # print(img.shape)
#                 img = my_augment_gamma(img, xy_resolution)
#                 # block reduce
#                 factor = 2
#                 img = block_reduce(img, block_size=(factor, factor, factor), func=np.max)
#                 tifffile.imwrite(os.path.join(imagests, target_name + '_0000.tif'), img.astype("uint8"))
#
#                 # spacing file!
#                 save_json({'spacing': spacing}, os.path.join(imagests, target_name + '.json'))
#         except Exception as e:
#             if (os.path.exists(os.path.join(imagests, target_name + '_0000.tif'))):
#                 os.remove(os.path.join(imagests, target_name + '_0000.tif'))
#             if (os.path.exists(os.path.join(imagests, target_name + '.json'))):
#                 os.remove(os.path.join(imagests, target_name + '.json'))
#             print(f"An error occurred: {e} at {im}")
#             # with open(f"error_log.txt", "a") as file:
#             #     file.write(f"An error occurred: {e} at {im}\n")
#
#         temp_im = im.split('/')[-1]
#         if (im.endswith('.v3draw')):
#             temp_im = temp_im[:-7]
#         data['full_name'].append(temp_im)
#         data['nnunet_name'].append(target_name)
#         data["spacing"].append(spacing)
#         data["img_size"].append(img_size)
#         data["ID"].append(int(id))
#         data["v3draw_path"].append(os.path.join(test_source, im))
#
#     df = pd.DataFrame(data)
#     df = df.sort_values(by='ID')
#     df.to_csv(origin_new_id_map_file, index=False)
#
#     progress_bar.close()
#
# test_source = long_3341_to_be_add_img_dir
# imagests = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/imagesTr"
# origin_new_id_map_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/long_590_test_data_for_nnunet/origin_new_id_map.csv"
# os.makedirs(imagests, exist_ok=True)
# print("Generating test data...")
# # 准备test data
# if(False):
#     generate_test_data(test_source, imagests, origin_new_id_map_file)
#
# #  test2

print("Generating recon mip...")
img_dir = long_3341_to_be_add_img_dir
seg_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/0_seg"
skel_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/3_skel_with_soma"
swc_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/6_connect_soma_swc"
# swc_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/8_estimated_radius_swc"
mip_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/recon_mip"
def generate_recon_mip(img_file, seg_file, skel_file, swc_file, mip_file):
    if(os.path.exists(mip_file)):
        return
    if(not os.path.exists(img_file) or not os.path.exists(seg_file) or not os.path.exists(skel_file) or not os.path.exists(swc_file)):
        return
    tif_file_list, swc_file_list, mip_list = [img_file, seg_file, skel_file,], [swc_file,], []
    for i, tif_file in enumerate(tif_file_list):
        tif = tifffile.imread(tif_file)
        if(i == 2):
            tif = np.flip(tif, axis=1)
            # print(tif.shape, type(tif))
        tif_mip = np.max(tif, axis=0).astype(np.float32)
        tif_mip = (tif_mip - np.min(tif_mip)) / (np.max(tif_mip) - np.min(tif_mip)) * 255
        mip_list.append(tif_mip)
    for swc_file in swc_file_list:
        swc_mip = get_mip_swc(swc_file, tif, ignore_background=True)
        mip_list.append(swc_mip)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(len(mip_list)):
        ax[i].imshow(mip_list[i], cmap='gray')
        ax[i].axis('off')
    plt.tight_layout()
    plt.savefig(mip_file)
    plt.close()

os.makedirs(mip_dir, exist_ok=True)
Parallel(n_jobs=8)(
    delayed(generate_recon_mip)(os.path.join(img_dir, os.path.basename(im).replace('.tif', '.tif')),
                                os.path.join(seg_dir, os.path.basename(im).replace('.tif', '.tif')),
                                os.path.join(skel_dir, os.path.basename(im).replace('.tif', '.tif')),
                                os.path.join(swc_dir, os.path.basename(im).replace('.tif', '.swc')),
                                os.path.join(mip_dir, os.path.basename(im).replace('.tif', '.png'))
                                )
    for im in tqdm(os.listdir(img_dir), desc="Generating recon mip")
)


