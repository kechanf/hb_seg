import os
import shutil
import subprocess
import uuid
from functools import partial
from multiprocessing import Pool

import cc3d
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage.morphology
import tifffile
from scipy.ndimage import binary_dilation
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops, label
from skimage.morphology import ball
from skimage.morphology import skeletonize_3d
from tqdm import tqdm

# os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"


import cupy as cp
from cupyx.scipy.ndimage import binary_opening
import cupyx

import SimpleITK as sitk
# from gcut.python.neuron_segmentation import NeuronSegmentation
import sys
from simple_swc_tool.swc_io import read_swc, write_swc

from scipy import ndimage
# from nnunetv2.training.loss.fastanison import anisodiff3
import pandas as pd

pool_num = 16


# set cuda_path


def maybe_mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


if (uuid.UUID(int=uuid.getnode()).hex[-12:] == "bc6ee23a04f7"):
    v3d_path = r"D:\Vaa3D-x.1.1.2_Windows_64bit\Vaa3D-x.exe"
elif (sys.platform == "linux"):
    v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"
# source_path = r"D:\tracing_ws\dataset\nnUNet_raw\Dataset150_human_brain_10000"
# raw_folder_path = os.path.join(source_path, "imagesTs")
# lab_folder_path = os.path.join(source_path, "labelsTr")
# spacing_folder_path = raw_folder_path
# name_mapping_path = os.path.join(source_path, "name_mapping.csv")
# name_mapping_path = r"C:\Users\12626\Desktop\name_mapping.csv"
# name_mapping_path = r"E:\tracing_ws\10847\name_mapping (2).csv"

# pred_path =  r"D:\tracing_ws\nnUNet\nnUNet_results\Dataset135_human_brain_10000"
# pred_path =  r"D:\tracing_ws\nnUNet\nnUNet_results\Dataset145_human_brain_10000_gamma"
# pred_path = pred_path+ r"\nnUNetTrainer__nnUNetPlans__3d_lowres\predicted_next_stage"
# pred_folder_path = os.path.join(pred_path, "3d_cascade_fullres")
# pred_path = r"D:\tracing_ws\nnUNet\nnUNet_results\150_test1223"
# pred_path = r"E:\tracing_ws\10847\TEST10K7"
data_source_folder_path = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset102_human_brain_test500"
result_folder_path = r"/data/kfchen/nnUNet/nnUNet_raw/result500_161_v13_e150"

trace_ws_path = r"/data/kfchen/trace_ws"
# make dir for new result folder
pred_path = os.path.join(trace_ws_path, str.split(result_folder_path, '/')[-1])
# print(pred_path)
maybe_mkdir_p(pred_path)
tif_folder_path = os.path.join(pred_path, "tif")
# copy every thing in result folder to pred folder
if(not os.path.exists(tif_folder_path) or len(os.listdir(tif_folder_path)) == 0):
    if(os.path.exists(tif_folder_path)):
        os.rmdir(tif_folder_path)
    shutil.copytree(result_folder_path, tif_folder_path)

comp_folder_path = os.path.join(pred_path, "comp")
skel_folder_path = os.path.join(pred_path, "skel")
clos_folder_path = os.path.join(pred_path, "clos")
swc_folder_path = os.path.join(pred_path, "swc")
soma_folder_path = os.path.join(pred_path, "soma")
preview_folder_path = os.path.join(pred_path, "preview")
skelwithsoma_folder_path = os.path.join(pred_path, "skelwithsoma")
somamarker_folder_path = os.path.join(pred_path, "somamarker")
uniswc_folder_path = os.path.join(pred_path, "uniswc")
conn_folder_path = os.path.join(pred_path, "connswc")
resample_folder_path = os.path.join(pred_path, "resampleswc")
v3dswc_folder_path = os.path.join(pred_path, "v3dswc")
soma_mip_folder_path = os.path.join(pred_path, "soma_mip")
adf_folder_path = os.path.join(pred_path, "adf")
list_traced_path = os.path.join(pred_path, "list_traced.xlsx")
gcutswc_folder_path = os.path.join(pred_path, "gcutswc")
gmsoma_marker_folder_path = os.path.join(pred_path, "gmsoma_markers")
# muti_soma_marker_folder_path = r"E:\tracing_ws\10847\muti_soma_markers"
muti_soma_marker_folder_path = r"/data/kfchen/trace_ws/muti_soma_markers"
name_mapping_path = os.path.join(pred_path, "name_mapping.csv")
if(not os.path.exists(name_mapping_path)):
    shutil.copy(os.path.join(data_source_folder_path, "name_mapping.csv"), name_mapping_path)
v3dswc_copy_folder_path = os.path.join(pred_path, "v3dswc_copy")

# pbd_folder_path = r"D:\tracing_ws\dataset\test1223"


maybe_mkdir_p(tif_folder_path)
maybe_mkdir_p(comp_folder_path)
maybe_mkdir_p(skel_folder_path)
maybe_mkdir_p(clos_folder_path)
maybe_mkdir_p(swc_folder_path)
maybe_mkdir_p(soma_folder_path)
maybe_mkdir_p(preview_folder_path)
maybe_mkdir_p(skelwithsoma_folder_path)
maybe_mkdir_p(somamarker_folder_path)
maybe_mkdir_p(uniswc_folder_path)
maybe_mkdir_p(conn_folder_path)
# maybe_mkdir_p(resample_folder_path)
maybe_mkdir_p(v3dswc_folder_path)
# maybe_mkdir_p(soma_mip_folder_path)
# maybe_mkdir_p(adf_folder_path)
# maybe_mkdir_p(gcutswc_folder_path)
maybe_mkdir_p(gmsoma_marker_folder_path)


def process_path(pstr):
    return pstr.replace('(', '\(').replace(')', '\)')


def remove_others_in_folder(folder_path):
    process_bar = tqdm(total=len(os.listdir(folder_path)), desc="remove_tif", unit="file")
    for file_name in os.listdir(folder_path):
        process_bar.update(1)
        if not file_name.endswith('.tif'):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
    process_bar.close()


def get_full_name(file_name, df):
    full_name = df[df['nnunet_name'] == file_name]['full_name'].values[0]
    return str(full_name)


def rename_tif_file(file_name, tif_folder_path, df):
    file_path = os.path.join(tif_folder_path, file_name)
    full_name = get_full_name(file_name.split('.')[0], df)
    new_file_path = os.path.join(tif_folder_path, full_name + '.tif')
    os.rename(file_path, new_file_path)


def rename_tif_folder(tif_folder):
    df = pd.read_csv(name_mapping_path)
    file_names = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    partial_func = partial(rename_tif_file, tif_folder_path=tif_folder, df=df)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="rename_tif_folder", unit="file"):
            pass


def uint8_tif_file(file_name, tif_folder_path):
    file_path = os.path.join(tif_folder_path, file_name)
    img = tifffile.imread(file_path)
    # normalize to 0-255
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    tifffile.imwrite(file_path, img)


def uint8_tif_folder(tif_folder):
    file_names = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    partial_func = partial(uint8_tif_file, tif_folder_path=tif_folder)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="uint8_tif_folder", unit="file"):
            pass


def skel_tif_file(file_name, tif_folder, skel_folder):
    tif_path = os.path.join(tif_folder, file_name)
    skel_path = os.path.join(skel_folder, os.path.splitext(file_name)[0] + '.tif')

    data = tifffile.imread(tif_path).astype(np.uint8)
    skel = skeletonize_3d(data).astype(np.uint8)
    # skel = binary_dilation(skel, iterations=1).astype(np.uint8)
    tifffile.imwrite(skel_path, skel * 255)


def skel_tif_folder(tif_folder, skel_folder):
    file_names = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    # partial_func = partial(skel_tif_file, tif_folder=tif_folder, skel_folder=skel_folder)
    # with Pool(pool_num) as p:
    #     for _ in tqdm(p.imap(partial_func, file_names),
    #                   total=len(file_names), desc="skel_tif_folder", unit="file"):
    #         pass
    fp_ratio_list = []
    for file_name in file_names:
        tiff = tifffile.imread(os.path.join(tif_folder, file_name))
        fp_ratio = np.sum(tiff) / 255
        fp_ratio = fp_ratio / (tiff.shape[0] * tiff.shape[1] * tiff.shape[2])
        fp_ratio_list.append(fp_ratio)

    print(f"mean fp ratio: {np.mean(fp_ratio_list)}")
    print(f"max fp ratio: {np.max(fp_ratio_list)}")
    print(f"min fp ratio: {np.min(fp_ratio_list)}")

    time.sleep(465456)




def dusting(img):
    if (img.sum() == 0):
        return img
    labeled_image = cc3d.connected_components(img, connectivity=6)
    largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    largest_component_binary = ((labeled_image == largest_label)).astype(np.uint8)
    return largest_component_binary


def get_min_diameter_3d(binary_image):
    labeled_array, num_features = label(binary_image)
    largest_cc = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    slice_x, slice_y, slice_z = find_objects(labeled_array == largest_cc)[0]
    diameter_x = slice_x.stop - slice_x.start
    diameter_y = slice_y.stop - slice_y.start
    diameter_z = slice_z.stop - slice_z.start

    return min(diameter_x, diameter_y, diameter_z)


def crop_nonzero(image):
    non_zero_coords = np.argwhere(image)
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0) + 1

    cropped_image = image[min_coords[0]:max_coords[0],
                    min_coords[1]:max_coords[1],
                    min_coords[2]:max_coords[2]]

    return cropped_image, image.shape, min_coords


def restore_original_size(cropped_image, original_shape, min_coords):
    restored_image = np.zeros(original_shape, dtype=cropped_image.dtype)
    restored_image[min_coords[0]:min_coords[0] + cropped_image.shape[0],
    min_coords[1]:min_coords[1] + cropped_image.shape[1],
    min_coords[2]:min_coords[2] + cropped_image.shape[2]] = cropped_image

    return restored_image


def opening_get_soma_region(soma_region):
    soma_region_copy = soma_region.copy()
    radius = get_min_diameter_3d(soma_region)

    # on cpu
    max_rate = 10
    for i in range(max_rate):
        spherical_selem = ball(radius * (max_rate - i) / 10 / 2)
        # soma_region_res = binary_opening(soma_region, spherical_selem).astype(np.uint8)
        soma_region_res = scipy.ndimage.binary_opening(soma_region, spherical_selem).astype(np.uint8)
        if (soma_region_res.sum() == 0):
            continue
        soma_region = soma_region_res

    # soma_region = binary_erosion(soma_region, spherical_selem).astype(np.uint8)
    del spherical_selem, radius, soma_region_res
    if (soma_region.sum() == 0):
        soma_region = soma_region_copy
    del soma_region_copy

    return soma_region


def opening_get_soma_region_gpu(soma_region):
    soma_region_copy = soma_region.copy()
    radius = get_min_diameter_3d(soma_region)

    # on gpu
    # try:
    max_rate = 10
    soma_region_gpu = cp.array(soma_region)

    for i in range(max_rate):
        spherical_selem = ball(radius * (max_rate - i) / 10 / 2)
        spherical_selem_gpu = cp.array(spherical_selem)

        # 在 GPU 上执行 binary_opening
        # soma_region_res_gpu = binary_opening(soma_region_gpu, spherical_selem_gpu)
        soma_region_res_gpu = cupyx.scipy.ndimage.binary_opening(soma_region_gpu, spherical_selem_gpu)

        if soma_region_res_gpu.sum() == 0:
            continue

        soma_region_gpu = soma_region_res_gpu

    soma_region = cp.asnumpy(soma_region_gpu)
    del spherical_selem, radius, soma_region_res_gpu, soma_region_gpu
    # except:
    #     pass
    if (soma_region.sum() == 0):
        soma_region = soma_region_copy
    del soma_region_copy

    return soma_region


def zero_outside_sphere(image, center, radius=25):
    z, y, x = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
    distance_from_center = np.sqrt((x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2)
    image[distance_from_center > radius] = 0

    # tifffile.imwrite(r"E:\tracing_ws\10847\TEST10K5\tif\222.tif", image)

    return image


def get_main_soma_region_in_msoma_from_gsdt(marker_path, gsdt, pred, in_tmp, out_tmp):
    marker_content = open(marker_path, 'r').read()
    # delete the first line
    if (marker_content.split('\n', 1)[0][0] == '#'):
        marker_content = marker_content.split('\n', 1)[1]
    x, y, z, _, _, _, _, _, _, _ = marker_content.split(',')
    x, y, z = float(x) / 2, float(y) / 2, float(z) / 2
    # print(x, y, z)
    gsdt = zero_outside_sphere(gsdt, (int(z), int(y), int(x)))
    if (gsdt.sum() == 0):
        if (os.path.exists(out_tmp)): os.remove(out_tmp)
        return None

    max_gsdt = np.max(gsdt)
    gsdt[gsdt <= max_gsdt / 2] = 0
    gsdt[gsdt > max_gsdt / 2] = 1

    gsdt = binary_dilation(gsdt, iterations=5).astype(np.uint8)
    soma_region = np.logical_and(pred, gsdt).astype(np.uint8)
    soma_region = dusting(soma_region)
    del pred, gsdt, max_gsdt

    soma_region, original_shape, min_coords = crop_nonzero(soma_region)
    # soma_region = opening_get_soma_region(soma_region)
    soma_region = opening_get_soma_region_gpu(soma_region)
    soma_region = dusting(soma_region)
    # restore original size
    soma_region = restore_original_size(soma_region, original_shape, min_coords)

    return soma_region


# def get_main_soma_region_in_msoma_from_soma_region(marker_path, soma_region):
#     marker_content = open(marker_path, 'r').read()
#     # delete the first line
#     if (marker_content.split('\n', 1)[0][0] == '#'):
#         marker_content = marker_content.split('\n', 1)[1]
#     x, y, z, _, _, _, _, _, _, _ = marker_content.split(',')
#     x, y, z = float(x) / 2, float(y) / 2, float(z) / 2
#     # print(x, y, z)


def remove_small_connected_blocks(img, min_size):
    labeled_img, num_features = ndimage.label(img)
    sizes = ndimage.sum(img, labeled_img, range(1, num_features + 1))
    mask_sizes = sizes >= min_size
    mask_sizes = np.concatenate(([False], mask_sizes))  # 为了使用布尔索引，加一个False对应背景
    labeled_img_clean = mask_sizes[labeled_img]

    # 重新标记以确保连通块编号连续
    labeled_img_clean, num_objects = ndimage.label(labeled_img_clean)

    return labeled_img_clean, num_objects


def get_msoma_region(gsdt, pred, marker_path):
    max_gsdt = np.max(gsdt)
    gsdt[gsdt <= max_gsdt / 2] = 0
    gsdt[gsdt > max_gsdt / 2] = 1

    gsdt = binary_dilation(gsdt, iterations=5).astype(np.uint8)
    soma_region = np.logical_and(pred, gsdt).astype(np.uint8)
    del pred, gsdt, max_gsdt

    min_size = 100
    while (True):
        labeled_img, num_objects = remove_small_connected_blocks(soma_region, min_size=min_size)
        if (num_objects <= 3):
            break
        else:
            min_size = min_size * 2

    result_img = np.zeros_like(soma_region)

    for obj_id in range(1, num_objects + 1):
        obj_img = np.where(labeled_img == obj_id, 1, 0)
        obj_img, original_shape, min_coords = crop_nonzero(obj_img)

        # soma_region = opening_get_soma_region(soma_region)
        obj_img = opening_get_soma_region_gpu(obj_img)
        obj_img = dusting(obj_img)
        # restore original size
        obj_img = restore_original_size(obj_img, original_shape, min_coords)

        while (np.sum(obj_img) < 100):
            obj_img = binary_dilation(obj_img, iterations=1).astype(np.uint8)

        result_img += obj_img
        del obj_img

    marker_content = open(marker_path, 'r').read()
    # delete the first line
    if (marker_content.split('\n', 1)[0][0] == '#'):
        marker_content = marker_content.split('\n', 1)[1]
    soma_x, soma_y, soma_z, _, _, _, _, _, _, _ = marker_content.split(',')
    soma_x, soma_y, soma_z = float(soma_x) / 2, float(soma_y) / 2, float(soma_z) / 2

    if (result_img[int(soma_z), int(soma_y), int(soma_x)] == 0):
        # create a ball
        ball = np.zeros_like(result_img)
        z, y, x = np.ogrid[:ball.shape[0], :ball.shape[1], :ball.shape[2]]
        distance_from_center = np.sqrt((x - int(soma_x)) ** 2 + (y - int(soma_y)) ** 2 + (z - int(soma_z)) ** 2)
        ball[distance_from_center <= 5] = 1
        result_img = np.logical_or(result_img, ball).astype(np.uint8)

    # binary
    result_img = np.where(result_img > 0, 1, 0)
    return result_img


def get_soma_region(img_path, marker_path=None):
    # print(img_path, marker_path)
    in_tmp = img_path
    out_tmp = in_tmp.replace('.tif', '_gsdt.tif')

    if (sys.platform == "linux"):
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x gsdt -f gsdt -i "{in_tmp}" -o "{out_tmp}" -p 0 1 0 1.5'
        cmd_str = process_path(cmd_str)
        print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
    else:
        cmd = f'{v3d_path} /x gsdt /f gsdt /i "{in_tmp}" /o "{out_tmp}" /p 0 1 0 1.5'
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    pred = tifffile.imread(img_path).astype(np.uint8)
    pred[pred <= 255 / 2] = 0
    pred[pred > 255 / 2] = 1

    gsdt = tifffile.imread(out_tmp).astype(np.uint8)
    gsdt = np.flip(gsdt, axis=1)
    if (os.path.exists(out_tmp)): os.remove(out_tmp)
    del out_tmp, in_tmp

    # save tif
    # tifffile.imwrite(r"E:\tracing_ws\10847\TEST10K5\tif\111.tif", gsdt)

    if (marker_path and os.path.exists(marker_path)):
        # return get_main_soma_region_in_msoma(marker_path, gsdt, pred, in_tmp, out_tmp)
        return get_msoma_region(gsdt, pred, marker_path)

    else:
        max_gsdt = np.max(gsdt)
        gsdt[gsdt <= max_gsdt / 2] = 0
        gsdt[gsdt > max_gsdt / 2] = 1

        gsdt = binary_dilation(gsdt, iterations=5).astype(np.uint8)
        soma_region = np.logical_and(pred, gsdt).astype(np.uint8)
        soma_region = dusting(soma_region)
        del pred, gsdt, max_gsdt

        soma_region, original_shape, min_coords = crop_nonzero(soma_region)
        # soma_region = opening_get_soma_region(soma_region)
        soma_region = opening_get_soma_region_gpu(soma_region)
        soma_region = dusting(soma_region)
        # restore original size
        soma_region = restore_original_size(soma_region, original_shape, min_coords)

        return soma_region


def find_muti_soma_marker_file(file_name, muti_soma_marker_folder):
    if (os.path.exists(os.path.join(muti_soma_marker_folder, os.path.splitext(file_name)[0] + '.marker'))):
        return os.path.join(muti_soma_marker_folder, os.path.splitext(file_name)[0] + '.marker')
    ID = file_name.split('_')[0]
    # find the soma marker according to the ID
    file_names = os.listdir(muti_soma_marker_folder)
    for file in file_names:
        if (file.split('_')[0] == ID):
            return os.path.join(muti_soma_marker_folder, file)
    return None


def get_soma_regions_file(file_name, tif_folder, soma_folder, muti_soma_marker_folder, do_mip=False):
    if ("gsdt" in file_name):
        os.remove(file_name)
        return
    tif_path = os.path.join(tif_folder, file_name)
    soma_region_path = os.path.join(soma_folder, os.path.splitext(file_name)[0] + '.tif')
    # muti_soma_marker_path = os.path.join(muti_soma_marker_folder, os.path.splitext(file_name)[0] + '.marker')
    muti_soma_marker_path = find_muti_soma_marker_file(file_name, muti_soma_marker_folder)

    if (os.path.exists(soma_region_path)):
        return
    try:
        soma_region = get_soma_region(tif_path, muti_soma_marker_path)
    except:
        print(f"error in {file_name}")
        return
    if (soma_region is None):
        return
    # binary
    soma_region = np.where(soma_region > 0, 1, 0).astype(np.uint8)
    tifffile.imwrite(soma_region_path, soma_region * 255)

    if (do_mip):
        soma_region_mip = np.max(soma_region, axis=0)
        soma_region_mip_path = os.path.join(soma_mip_folder_path, os.path.splitext(file_name)[0] + '.tif')
        tifffile.imwrite(soma_region_mip_path, soma_region_mip * 255)

    del soma_region


def get_soma_regions_folder(tif_folder, soma_folder, muti_soma_marker_folder, time_out=60):
    file_names = os.listdir(tif_folder)
    for file_name in file_names:
        if ("gsdt" in str(file_name)):
            os.remove(os.path.join(tif_folder, file_name))

    file_names = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    partial_func = partial(get_soma_regions_file, tif_folder=tif_folder, soma_folder=soma_folder,
                           muti_soma_marker_folder=muti_soma_marker_folder)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="get_soma_regions_folder", unit="file"):
            pass

    file_names = os.listdir(tif_folder)
    for file_name in file_names:
        if ("gsdt" in str(file_name)):
            os.remove(os.path.join(tif_folder, file_name))


def compute_centroid(mask):
    # 计算三维 mask 的重心
    labeled_mask = skimage.measure.label(mask)
    props = regionprops(labeled_mask)

    if len(props) > 0:
        # 获取第一个区域的重心坐标
        centroid = props[0].centroid
        return centroid
    else:
        return None


def get_somamarker_file(file_name, soma_folder, somamarker_folder, muti_soma_marker_folder, gm_soma_marker_folder):
    soma_path = os.path.join(soma_folder, file_name)
    somamarker_path = os.path.join(somamarker_folder, os.path.splitext(file_name)[0] + '.marker')
    gmsoma_marker_folder_path = os.path.join(gm_soma_marker_folder, os.path.splitext(file_name)[0] + '.marker')
    muti_soma_marker_path = find_muti_soma_marker_file(file_name, muti_soma_marker_folder)

    if (os.path.exists(somamarker_path)):
        return

    if (muti_soma_marker_path and os.path.exists(muti_soma_marker_path)):
        # shutil.copy(muti_soma_marker_path, somamarker_path)
        # open soma marker file
        with open(muti_soma_marker_path, 'r') as f:
            marker_content = f.read()
            # delete the first line
            if (marker_content.split('\n', 1)[0][0] == '#'):
                marker_content = marker_content.split('\n', 1)[1]
            marker_content = marker_content.split(',')
            label_soma_x, label_soma_y, label_soma_z = float(marker_content[0]), float(marker_content[1]), float(
                marker_content[2])
            label_soma_x, label_soma_y, label_soma_z = int(label_soma_x / 2), int(label_soma_y / 2), int(
                label_soma_z / 2)

            soma_region = tifffile.imread(soma_path).astype(np.uint8)
            labeled_img, num_objects = ndimage.label(soma_region)

            min_dis = 100000
            main_soma_x, main_soma_y, main_soma_z = 0, 0, 0

            soma_list = []

            for obj_id in range(1, num_objects + 1):
                obj_img = np.where(labeled_img == obj_id, 1, 0)
                centroid = compute_centroid(obj_img)
                soma_x, soma_y, soma_z = centroid[2], centroid[1], centroid[0]
                soma_y = soma_region.shape[1] - soma_y - 1
                soma_list.append([soma_x, soma_y, soma_z])

                dist_to_label = (soma_x - label_soma_x) ** 2 + (soma_y - label_soma_y) ** 2 + (
                            soma_z - label_soma_z) ** 2
                if (dist_to_label < min_dis):
                    min_dis = dist_to_label
                    main_soma_x, main_soma_y, main_soma_z = soma_x, soma_y, soma_z
                del obj_img, centroid, soma_x, soma_y, soma_z, dist_to_label

            marker_str = f"{main_soma_x}, {main_soma_y}, {main_soma_z}, 1, 1, , , 255,0,0\n"
            with open(somamarker_path, 'w') as ff:
                ff.write(marker_str)

            for soma in soma_list:
                if (soma[0] == main_soma_x and soma[1] == main_soma_y and soma[2] == main_soma_z):
                    continue
                marker_str += f"{soma[0]}, {soma[1]}, {soma[2]}, 1, 1, , , 0,255,0\n"
            with open(gmsoma_marker_folder_path, 'w') as ff:
                ff.write(marker_str)

        return
    else:
        soma_region = tifffile.imread(soma_path).astype(np.uint8)
        centroid = compute_centroid(soma_region)
        soma_x, soma_y, soma_z, soma_r = centroid[2], centroid[1], centroid[0], 1
        soma_y = soma_region.shape[1] - soma_y
        # print(soma_x, soma_y, soma_z, soma_r)
        marker_str = f"{soma_x}, {soma_y}, {soma_z}, {soma_r}, 1, , , 255,0,0"
        with open(somamarker_path, 'w') as f:
            f.write(marker_str)


def get_somamarker_folder(soma_folder, somamarker_folder, muti_soma_marker_folder, gm_soma_marker_folder):
    file_names = [f for f in os.listdir(soma_folder) if f.endswith('.tif')]
    partial_func = partial(get_somamarker_file, soma_folder=soma_folder, somamarker_folder=somamarker_folder,
                           muti_soma_marker_folder=muti_soma_marker_folder, gm_soma_marker_folder=gm_soma_marker_folder)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="get_somamarker_folder", unit="file"):
            pass


def get_skelwithsoma_file(file_name, skel_folder, soma_folder, skelwithsoma_folder):
    skel_path = os.path.join(skel_folder, file_name)
    soma_path = os.path.join(soma_folder, file_name)
    skelwithsoma_path = os.path.join(skelwithsoma_folder, os.path.splitext(file_name)[0] + '.tif')

    if (os.path.exists(skelwithsoma_path)):
        return
    if (not os.path.exists(soma_path) or not os.path.exists(skel_path)):
        return

    skel = tifffile.imread(skel_path).astype(np.uint8)
    soma = tifffile.imread(soma_path).astype(np.uint8)
    skelwithsoma = (np.logical_or(skel, soma))
    # normalize to [0, 255]
    skelwithsoma = (skelwithsoma * 255).astype(np.uint8)
    tifffile.imwrite(skelwithsoma_path, skelwithsoma)


def get_skelwithsoma_folder(skel_folder, soma_folder, skelwithsoma_folder):
    file_names = [f for f in os.listdir(skel_folder) if f.endswith('.tif')]
    partial_func = partial(get_skelwithsoma_file, skel_folder=skel_folder, soma_folder=soma_folder,
                           skelwithsoma_folder=skelwithsoma_folder)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="get_skelwithsoma_folder", unit="file"):
            pass


def calc_connected_block_num(img):
    labeled_img, num_objects = ndimage.label(img)
    return num_objects


# Anisotropic Diffusion Filtering
def adf_file(file_name, tif_folder, adf_folder, num_iterations, time_step, conductance):
    """
        三维图像的各向异性扩散滤波。

        Parameters:
        - num_iterations: int, 迭代次数。
        - time_step: float, 时间步长。
        - conductance: float, 导电系数，控制扩散的程度。

    """
    tif_path = os.path.join(tif_folder, file_name)
    adf_path = os.path.join(adf_folder, file_name)

    input_image = sitk.ReadImage(tif_path, sitk.sitkFloat32)

    # 创建各向异性扩散滤波器
    filter_ = sitk.CurvatureAnisotropicDiffusionImageFilter()
    filter_.SetNumberOfIterations(num_iterations)
    filter_.SetTimeStep(time_step)
    filter_.SetConductanceParameter(conductance)

    while (True):
        filtered_image = filter_.Execute(input_image)
        diff_image = sitk.GetArrayFromImage(filtered_image) - sitk.GetArrayFromImage(input_image)
        if (np.max(diff_image) < 0.1 and conductance < 2):
            conductance = conductance + 0.1
            filter_.SetConductanceParameter(conductance)
        else:
            break

    # binary
    filtered_image = sitk.GetArrayFromImage(filtered_image)
    filtered_image = np.where(filtered_image > 0, 1, 0).astype(np.uint8) * 255

    tifffile.imwrite(adf_path, filtered_image)
    # sitk.WriteImage(filtered_image, adf_path)

    # compare with the original image
    original_image = tifffile.imread(tif_path).astype(np.float32)
    filtered_image = tifffile.imread(adf_path).astype(np.float32)
    # print(f"max: {np.max(original_image)}, min: {np.min(original_image)}"
    # f"max: {np.max(filtered_image)}, min: {np.min(filtered_image)}")
    diff_image = original_image - filtered_image
    # print(f"max: {np.max(diff_image)}, min: {np.min(diff_image)}")
    diff_image = np.where(diff_image > 0, 1, diff_image)
    diff_image = np.where(diff_image < 0, 1, diff_image)
    # print(f"max: {np.max(diff_image)}, min: {np.min(diff_image)}")
    # normalize to [0, 255]
    diff_image = (diff_image / np.max(diff_image) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    axes[0].imshow(np.max(original_image, axis=0), cmap='gray')
    axes[0].set_title(f'Original, num: {calc_connected_block_num(original_image)}')
    axes[1].imshow(np.max(filtered_image, axis=0), cmap='gray')
    axes[1].set_title(f'Filtered, num: {calc_connected_block_num(filtered_image)}')
    axes[2].imshow(np.max(diff_image, axis=0), cmap='gray')
    axes[2].set_title('Difference')
    plt.savefig(adf_path.replace('.tif', '.png'))


#
# def adf_file_fastaniso(file_name, tif_folder, adf_folder):
#     tif_path = os.path.join(tif_folder, file_name)
#     adf_path = os.path.join(adf_folder, file_name)
#
#     input_image = tifffile.imread(tif_path).astype(np.float32)
#     filtered_image = anisodiff3(input_image, niter=1, kappa=50, gamma=0.1, option=1)
#     filtered_image = np.where(filtered_image > 0, 1, 0).astype(np.uint8) * 255
#
#     tifffile.imwrite(adf_path, filtered_image)
#     original_image = tifffile.imread(tif_path).astype(np.float32)
#     filtered_image = tifffile.imread(adf_path).astype(np.float32)
#     diff_image = original_image - filtered_image
#     diff_image = np.where(diff_image > 0, 1, diff_image)
#     diff_image = np.where(diff_image < 0, 1, diff_image)
#     diff_image = (diff_image / np.max(diff_image) * 255).astype(np.uint8)
#     fig, axes = plt.subplots(1, 3, figsize=(8, 4))
#     axes[0].imshow(np.max(original_image, axis=0), cmap='gray')
#     axes[0].set_title(f'Original, num: {calc_connected_block_num(original_image)}')
#     axes[1].imshow(np.max(filtered_image, axis=0), cmap='gray')
#     axes[1].set_title(f'Filtered, num: {calc_connected_block_num(filtered_image)}')
#     axes[2].imshow(np.max(diff_image, axis=0), cmap='gray')
#     axes[2].set_title('Difference')
#     plt.savefig(adf_path.replace('.tif', '.png'))
#

# def adf_folder(tif_folder, adf_folder, num_iterations=20, time_step=0.000625, conductance=1.1):
#     file_names = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
#     partial_func = partial(adf_file_fastaniso, tif_folder=tif_folder, adf_folder=adf_folder)
#     # partial_func = partial(adf_file, tif_folder=tif_folder, adf_folder=adf_folder, num_iterations=num_iterations, time_step=time_step, conductance=conductance)
#     with Pool(pool_num) as p:
#         for _ in tqdm(p.imap(partial_func, file_names),
#                       total=len(file_names), desc="adf_folder", unit="file"):
#             pass

def trace_app2_with_soma_file(file_name, skelwithsoma_folder, somamarker_folder, swc_folder):
    skelwithsoma_path = os.path.join(skelwithsoma_folder, file_name)
    somamarker_path = os.path.join(somamarker_folder, os.path.splitext(file_name)[0] + '.marker')
    swc_path = os.path.join(swc_folder, os.path.splitext(file_name)[0] + '.swc')
    ini_swc_path = skelwithsoma_path.replace('.tif', '.tif_ini.swc')
    if (os.path.exists(swc_path)):
        return
    '''
        **** Usage of APP2 ****
        vaa3d -x plugin_name -f app2 -i <inimg_file> -o <outswc_file> -p [<inmarker_file> [<channel> [<bkg_thresh> 
        [<b_256cube> [<b_RadiusFrom2D> [<is_gsdt> [<is_gap> [<length_thresh> [is_resample][is_brightfield][is_high_intensity]]]]]]]]]
        inimg_file          Should be 8/16/32bit image
        inmarker_file       If no input marker file, please set this para to NULL and it will detect soma automatically.
                            When the file is set, then the first marker is used as root/soma.
        channel             Data channel for tracing. Start from 0 (default 0).
        bkg_thresh          Default 10 (is specified as AUTO then auto-thresolding)
        b_256cube           If trace in a auto-downsampled volume (1 for yes, and 0 for no. Default 1.)
        b_RadiusFrom2D      If estimate the radius of each reconstruction node from 2D plane only (1 for yes as many 
        times the data is anisotropic, and 0 for no. Default 1 which which uses 2D estimation.)
        is_gsdt             If use gray-scale distance transform (1 for yes and 0 for no. Default 0.)
                       If allow gap (1 for yes and 0 for no. Default 0.)
        length_thresh       Default 5
        is_resample         If allow resample (1 for yes and 0 for no. Default 1.)
        is_brightfield      If the signals are dark instead of bright (1 for yes and 0 for no. Default 0.)
        is_high_intensity   If the image has high intensity background (1 for yes and 0 for no. Default 0.)
        outswc_file         If not be specified, will be named automatically based on the input image file name.
    '''

    resample = 1
    gsdt = 1
    b_RadiusFrom2D = 1

    if (not os.path.exists(somamarker_path)):
        somamarker_path = "NULL"

    # try:
    if (sys.platform == "linux"):
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x vn2 -f app2 -i {skelwithsoma_path} -o {swc_path} -p {somamarker_path} 0 10 1 {b_RadiusFrom2D} {gsdt} 1 5 {resample} 0 0'
        cmd = process_path(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
    else:
        vn2_path = r"E:/tracing_ws/vn2.dll"
        cmd = f'{v3d_path} /x {vn2_path} /f app2 /i {skelwithsoma_path} /o {swc_path} /p {somamarker_path} 0 10 1 1 {gsdt} 1 5 {resample} 0 0'
        # print(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # except:
    #     print("error at ", skelwithsoma_path)

    if (os.path.exists(ini_swc_path)):
        os.remove(ini_swc_path)


def trace_app2_with_soma_folder(skelwithsoma_folder, somamarker_folder, swc_folder):
    file_names = [f for f in os.listdir(skelwithsoma_folder) if f.endswith('.tif')]
    partial_func = partial(trace_app2_with_soma_file, skelwithsoma_folder=skelwithsoma_folder,
                           somamarker_folder=somamarker_folder, swc_folder=swc_folder)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="trace_app2_with_soma_folder", unit="file"):
            pass

    ini_file_names = [f for f in os.listdir(skelwithsoma_folder) if "ini" in f]
    for file_name in ini_file_names:
        os.remove(os.path.join(skelwithsoma_folder, file_name))


def trace_app1_with_soma_file(file_name, skelwithsoma_folder, somamarker_folder, swc_folder):
    """
    **** Usage of APP1 ****
    vaa3d -x plugin_name -f app1 -i <inimg_file> -p [<inmarker_file> [<channel> [<bkg_thresh> [<b_256cube> ]]]]
    inimg_file       Should be 8/16/32bit image
    inmarker_file    If no input marker file, please set this para to NULL and it will detect soma automatically.
                     When the file is set, then the first marker is used as root/soma.
    channel          Data channel for tracing. Start from 0 (default 0).
    bkg_thresh       Default AUTO (AUTO is for auto-thresholding), otherwise the threshold specified by a user will be used.
    b_256cube        If trace in a auto-downsampled volume (1 for yes, and 0 for no. Default 1.)
    outswc_file      If not be specified, will be named automatically based on the input image file name.
    """
    skelwithsoma_path = os.path.join(skelwithsoma_folder, file_name)
    somamarker_path = os.path.join(somamarker_folder, os.path.splitext(file_name)[0] + '.marker')
    swc_path = os.path.join(swc_folder, os.path.splitext(file_name)[0] + '.swc')
    ini_swc_path = skelwithsoma_path.replace('.tif', '.tif_ini.swc')
    if (os.path.exists(swc_path)):
        return

    resample = 1
    gsdt = 0
    b_RadiusFrom2D = 1

    if (not os.path.exists(somamarker_path)):
        somamarker_path = "NULL"

    try:
        if (sys.platform == "linux"):
            cmd = "???"
            # cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x vn2 -f app1 -i {skelwithsoma_path} -o {swc_path} -p {somamarker_path} 0 AUTO 1 {b_RadiusFrom2D} {gsdt} 1 5 {resample} 0 0'
            cmd = process_path(cmd)
            subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
        else:
            cmd = f'{v3d_path} /x D:/tracing_ws/vn2.dll /f app1 /i {skelwithsoma_path} /o {swc_path} /p {somamarker_path} 0 10 1'
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except:
        print("error at ", skelwithsoma_path)

    if (os.path.exists(ini_swc_path)):
        os.remove(ini_swc_path)


def trace_app1_with_soma_folder(skelwithsoma_folder, somamarker_folder, swc_folder):
    file_names = [f for f in os.listdir(skelwithsoma_folder) if f.endswith('.tif')]
    partial_func = partial(trace_app1_with_soma_file, skelwithsoma_folder=skelwithsoma_folder,
                           somamarker_folder=somamarker_folder, swc_folder=swc_folder)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="trace_app1_with_soma_folder", unit="file"):
            pass


#
# def gcut_file(file_name, name_mapping, swc_folder, gmsoma_marker_folder, gcut_folder):
#     swc_path = os.path.join(swc_folder, file_name)
#     gmsoma_marker_path = os.path.join(gmsoma_marker_folder, os.path.splitext(file_name)[0] + '.marker')
#     gcutswc_path = os.path.join(gcut_folder, file_name)
#
#     ID = int(file_name.split('_')[0])
#     spacing = name_mapping[name_mapping["ID"] == ID]["spacing"].values[0]
#     spacing = float(spacing.split(',')[1])
#
#     point_l = read_swc(swc_path)
#     if(len(point_l.p) <= 1):
#         return
#     point_l.p[1].x, point_l.p[1].y, point_l.p[1].z = int(point_l.p[1].x), int(point_l.p[1].y), int(point_l.p[1].z)
#     # if(os.path.exists(swc_path)):
#     #     os.remove(swc_path)
#
#
#     with open(gmsoma_marker_path, 'r') as f:
#         marker_content = f.read()
#         res_marker_content = marker_content
#         if(marker_content.split('\n', 1)[0][0] == '#'):
#             marker_content = marker_content.split('\n', 1)[1]
#         for line in marker_content.split('\n'):
#             if(line == ""):
#                 continue
#             line_content = line.split(',')
#             x, y, z = float(line_content[0]), float(line_content[1]), float(line_content[2])
#             res_x, res_y, res_z = x, y, z
#             min_dist = 100000
#             for p in point_l.p:
#                 dist = (p.x - x) ** 2 + (p.y - y) ** 2 + (p.z - z) ** 2
#                 if(dist < 100 and p.p == -1):
#                     res_x, res_y, res_z = int(p.x), int(p.y), int(p.z)
#                     # point_l.p[p.n].x, point_l.p[p.n].y, point_l.p[p.n].z = int(res_x), int(res_y), int(res_z)
#                     break
#                 if(dist < min_dist):
#                     min_dist = dist
#                     res_x, res_y, res_z = int(p.x), int(p.y), int(p.z)
#             # for p in point_l.p:
#             #     if(p.x == res_x and p.y == res_y and p.z == res_z):
#             #         res_x, res_y, res_z = int(res_x), int(res_y), int(res_z)
#             #         point_l.p[p.n].x, point_l.p[p.n].y, point_l.p[p.n].z = int(res_x), int(res_y), int(res_z)
#             #         break
#
#             res_marker_content += f"{res_x}, {res_y}, {res_z}, 1, 1, , , 255,0,0\n"
#     # if(os.path.exists(gmsoma_marker_path)):
#     #     os.remove(gmsoma_marker_path)
#     with open(gmsoma_marker_path, 'w') as f:
#         # print(res_marker_content)
#         f.write(res_marker_content)
#
#     # Writeswc_v2(swc_path, point_l)
#
#     # print(swc_path, somamarker_path, gcutswc_path, spacing)
#     try:
#         segmentor = NeuronSegmentation(swc_path, gmsoma_marker_path, scale_z=spacing, scale_ouput_z=False)
#         segmentor.segment()
#         gcutswc_path = gcutswc_path[:-4]
#         segmentor.save(gcutswc_path)
#
#         if os.path.isdir(gcutswc_path):
#             files = [f for f in os.listdir(gcutswc_path) if os.path.isfile(os.path.join(gcutswc_path, f))]
#             # sort
#             files.sort()
#
#             original_file_path = os.path.join(gcutswc_path, files[0])
#             new_file_path = os.path.join(gcut_folder, file_name)
#             if(r"soma=1" in original_file_path):
#                 shutil.copy(original_file_path, new_file_path)
#             shutil.rmtree(gcutswc_path)
#     except:
#         pass
#
#
# def gcut_folder(swc_folder, gmsoma_marker_folder, gcutswc_folder):
#     file_names = [f for f in os.listdir(swc_folder) if f.endswith('.swc')]
#     name_mapping = pd.read_csv(name_mapping_path)
#
#     partial_func = partial(gcut_file, name_mapping=name_mapping, swc_folder=swc_folder,
#                            gmsoma_marker_folder=gmsoma_marker_folder, gcut_folder=gcutswc_folder)
#     with Pool(pool_num) as p:
#         for _ in tqdm(p.imap(partial_func, file_names),
#                       total=len(file_names), desc="gcut_folder", unit="file"):
#             pass
#
#     # path = gcutswc_folder_path
#     # for item in os.listdir(path):
#     #     item_path = os.path.join(path, item)
#     #     # 检查是否为目录
#     #     if os.path.isdir(item_path):
#     #         files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
#     #         # 检查目录中是否只有一个文件
#     #         if len(files) == 1:
#     #             # 构建原文件的完整路径和目标文件的完整路径
#     #             original_file_path = os.path.join(item_path, files[0])
#     #             new_file_path = os.path.join(path, item + os.path.splitext(files[0])[1])
#     #             # 移动并重命名文件
#     #             shutil.move(original_file_path, new_file_path)
#     #             # 删除原目录
#     #             os.rmdir(item_path)

def prune_fiber_in_soma(point_l, soma_region):
    edge_p_list = []
    x_limit, y_limit, z_limit = soma_region.shape[2], soma_region.shape[1], soma_region.shape[0]

    for p in point_l.p:
        if (p.n == 0 or p.n == 1): continue
        x, y, z = p.x, p.y, p.z
        y = soma_region.shape[1] - y

        x = min(int(x), x_limit - 1)
        y = min(int(y), y_limit - 1)
        z = min(int(z), z_limit - 1)

        if (soma_region[int(z), int(y), int(x)]):
            edge_p_list.append(p)

    for p in edge_p_list:
        if (len(p.s) == 0):
            temp_p = point_l.p[p.n]
            while (True):
                if (temp_p.n == 1): break
                if (temp_p.pruned == True): break
                if (not len(temp_p.s) == 1): break
                point_l.p[temp_p.n].pruned = True
                temp_p = point_l.p[temp_p.p]
    for p in point_l.p:
        for s in p.s:
            if (point_l.p[s].pruned == True):
                p.s.remove(s)

    return point_l


def connect_to_soma_file(file_name, swc_folder, soma_folder, conn_folder):
    soma_region_path = os.path.join(soma_folder, os.path.splitext(file_name)[0] + '.tif')
    swc_path = os.path.join(swc_folder, os.path.splitext(file_name)[0] + '.swc')
    conn_path = os.path.join(conn_folder, os.path.splitext(file_name)[0] + '.swc')

    soma_region = tifffile.imread(soma_region_path).astype(np.uint8)
    # soma_region = get_main_soma_region_in_msoma_from_gsdt(soma_region,,
    soma_region = binary_dilation(soma_region, iterations=4).astype(np.uint8)
    x_limit, y_limit, z_limit = soma_region.shape[2], soma_region.shape[1], soma_region.shape[0]

    point_l = read_swc(swc_path)

    labeled_img, num_objects = ndimage.label(soma_region)
    if (len(point_l.p) <= 1):
        return
    for obj_id in range(1, num_objects + 1):
        obj_img = np.where(labeled_img == obj_id, 1, 0)
        x, y, z = point_l.p[1].x, point_l.p[1].y, point_l.p[1].z
        if (obj_img[int(z), int(y), int(x)]):
            soma_region = obj_img
            del obj_img
            break
        del obj_img
    # tifffile.imwrite(os.path.join(conn_folder, file_name+"1.tif"), soma_region.astype(np.uint8)*255)
    labeled_img, num_objects = ndimage.label(soma_region)
    if (num_objects > 1):
        # soma_region = dusting(soma_region)
        write_swc(conn_path, point_l)
        del soma_region, point_l
        return

    point_l = prune_fiber_in_soma(point_l, soma_region)

    # strict strategy
    edge_p_list = []
    for p in point_l.p:
        if (p.n == 0 or p.n == 1): continue
        x, y, z = p.x, p.y, p.z
        y = soma_region.shape[1] - y

        x = min(int(x), x_limit - 1)
        y = min(int(y), y_limit - 1)
        z = min(int(z), z_limit - 1)

        if (soma_region[int(z), int(y), int(x)]):
            edge_p_list.append(p)

    for p in edge_p_list:
        temp_p = point_l.p[p.p]
        while (True):
            if (temp_p.n == 1): break
            if (temp_p.pruned == True): break
            point_l.p[temp_p.n].pruned = True
            temp_p = point_l.p[temp_p.p]

    for p in edge_p_list:
        if (point_l.p[p.n].pruned == False):
            if (not len(point_l.p[p.n].s)):
                point_l.p[p.n].pruned = True
            else:
                point_l.p[p.n].p = 1
                point_l.p[1].s.append(p.n)
        else:
            for s in point_l.p[p.n].s:
                point_l.p[s].p = 1
                point_l.p[1].s.append(s)

    # Conservative strategy
    # for s in point_l.p[1].s:
    #     temp_p = point_l.p[s]
    #     x, y, z = temp_p.x, temp_p.y, temp_p.z
    #     y = soma_region.shape[1] - y
    #     x = min(int(x), x_limit - 1)
    #     y = min(int(y), y_limit - 1)
    #     z = min(int(z), z_limit - 1)
    #
    #     if(not soma_region[int(z), int(y), int(x)]):
    #         continue
    #
    #     for s2 in point_l.p[s].s:
    #         point_l.p[s2].p = 1
    #         point_l.p[1].s.append(s2)
    #
    #     point_l.p[1].s.remove(s)
    #     point_l.p[s].pruned = True

    write_swc(conn_path, point_l)
    del soma_region, point_l


def connect_to_soma_folder(swc_folder, soma_folder, conn_folder):
    file_names = [f for f in os.listdir(swc_folder) if f.endswith('.swc')]
    partial_func = partial(connect_to_soma_file, swc_folder=swc_folder, soma_folder=soma_folder,
                           conn_folder=conn_folder)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="connect_to_soma_folder", unit="file"):
            pass


def to_v3dswc_file(file_name, swc_folder, v3dswc_folder):
    swc_path = os.path.join(swc_folder, file_name)
    v3dswc_path = os.path.join(v3dswc_folder, os.path.splitext(file_name)[0] + '.swc')

    # try:
    # print(os.path.basename(swc_path).replace('.swc', ''))
    # origin_name = get_full_name(os.path.basename(swc_path).replace('.swc', ''))
    full_name = os.path.basename(swc_path).replace('.swc', '')
    # print(full_name)
    df = pd.read_csv(name_mapping_path)
    img_size = df[df['full_name'] == full_name]['img_size'].values[0]
    # print(img_size)
    img_size = img_size.split(',')

    x_limit, y_limit, z_limit = img_size[2], img_size[1], img_size[0]
    x_limit, y_limit, z_limit = "".join(filter(str.isdigit, x_limit)), \
        "".join(filter(str.isdigit, y_limit)), \
        "".join(filter(str.isdigit, z_limit))
    x_limit, y_limit, z_limit = int(x_limit), int(y_limit), int(z_limit)
    # print(x_limit, y_limit, z_limit)

    # except:
    #     try:
    #         if (raw_path.endswith('.tif')):
    #             img = tifffile.imread(raw_path)
    #             x_limit, y_limit, z_limit = img.shape[2], img.shape[1], img.shape[0]
    #             x_limit, y_limit, z_limit = x_limit * 2, y_limit * 2, z_limit * 2
    #     except:
    #         x_limit, y_limit, z_limit = 512, 512, 256
    #         pass

    with open(swc_path, 'r') as f:
        lines = f.readlines()
    res_lines = []
    # print(x_limit, y_limit, z_limit)
    for line in lines:
        if (line[0] == '#'): continue
        temp_line = line.split()

        x, y, z = float(temp_line[2]) * 2, float(temp_line[3]) * 2, float(temp_line[4]) * 2
        y = y_limit - y - 1
        #
        x, y, z = max(x, 0), max(y, 0), max(z, 0)
        # print(x, x_limit, y, y_limit, z, z_limit)
        x, y, z = min(x, x_limit - 1), min(y, y_limit - 1), min(z, z_limit - 1)

        res_line = "%s 247 %s %s %s %s %s\n" % (
            temp_line[0], str(x), str(y), str(z), str(temp_line[5]), temp_line[6]
        )
        res_lines.append(res_line)

    file_handle = open(v3dswc_path, mode="a")
    file_handle.writelines(res_lines)
    file_handle.close()


def to_v3dswc_folder(swc_folder, v3dswc_folder):
    file_names = [f for f in os.listdir(swc_folder) if f.endswith('.swc')]
    partial_func = partial(to_v3dswc_file, swc_folder=swc_folder, v3dswc_folder=v3dswc_folder)
    with Pool(pool_num) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="to_v3dswc_folder", unit="file"):
            pass


def get_list_traced(swc_folder, list_traced_path):
    # get swc files
    swc_files = [f for f in os.listdir(swc_folder) if f.endswith('.swc')]

    # 对文件名进行排序（如果需要）
    swc_files.sort()
    swc_ID = [int(i.split('_')[0]) for i in swc_files]

    # 创建一个 DataFrame 来存储文件名
    df = pd.DataFrame(swc_files, columns=['full_name'])
    df['ID'] = swc_ID

    # 保存 DataFrame 到 Excel 文件
    df.to_excel(list_traced_path, index=False)
    duplicate_ids = df['ID'].duplicated().any()
    if duplicate_ids:
        print("Warning: 有重复的 ID。")
        print(df[df['ID'].duplicated()]['ID'].values)
    else:
        print("所有 ID 均不重复。")

    print(f"文件已保存为: {list_traced_path}")


def compare_tif(folder1, folder2, out_folder):
    file_names1 = [f for f in os.listdir(folder1) if f.endswith('.tif')]
    file_names2 = [f for f in os.listdir(folder2) if f.endswith('.tif')]
    for file_name1 in file_names1:
        if (file_name1 not in file_names2):
            continue
        img1 = tifffile.imread(os.path.join(folder1, file_name1))
        img2 = tifffile.imread(os.path.join(folder2, file_name1))

        mip1 = np.max(img1, axis=0)
        mip2 = np.max(img2, axis=0)

        comp_mip = np.concatenate((mip1, mip2), axis=1)
        tifffile.imsave(os.path.join(out_folder, file_name1), comp_mip)


def prepossessing():
    remove_others_in_folder(tif_folder_path)
    rename_tif_folder(tif_folder_path)
    uint8_tif_folder(tif_folder_path)
    #
    # # ###########adf_folder(tif_folder_path, adf_folder_path)
    #
    skel_tif_folder(tif_folder_path, skel_folder_path)
    get_soma_regions_folder(tif_folder_path, soma_folder_path, muti_soma_marker_folder_path)
    get_skelwithsoma_folder(skel_folder_path, soma_folder_path, skelwithsoma_folder_path)
    get_somamarker_folder(soma_folder_path, somamarker_folder_path, muti_soma_marker_folder_path,
                          gmsoma_marker_folder_path)
    pass


def temp_prepossessing():
    num = 0
    file_names = os.listdir(tif_folder_path)
    for file_name in file_names:
        soma_path = os.path.join(soma_folder_path, file_name)
        somamarker_path = os.path.join(somamarker_folder_path, os.path.splitext(file_name)[0] + '.marker')
        # muti_soma_marker_path = os.path.join(muti_soma_marker_folder_path, os.path.splitext(file_name)[0] + '.marker')
        muti_soma_marker_path = find_muti_soma_marker_file(file_name, muti_soma_marker_folder_path)
        skelwithsoma_path = os.path.join(skelwithsoma_folder_path, os.path.splitext(file_name)[0] + '.tif')
        if (muti_soma_marker_path and os.path.exists(muti_soma_marker_path)):
            if (os.path.exists(somamarker_path)):
                os.remove(somamarker_path)
            if (os.path.exists(soma_path)):
                os.remove(soma_path)
            if (os.path.exists(skelwithsoma_path)):
                os.remove(skelwithsoma_path)
            num += 1
    print(num)


def tracing():
    trace_app2_with_soma_folder(skelwithsoma_folder_path, somamarker_folder_path, swc_folder_path)
    # trace_app1_with_soma_folder(skelwithsoma_folder_path, somamarker_folder_path, swc_folder_path)
    pass


def postprocessing():
    # gcut_folder(swc_folder_path, gmsoma_marker_folder_path, gcutswc_folder_path)
    connect_to_soma_folder(swc_folder_path, soma_folder_path, conn_folder_path)
    to_v3dswc_folder(conn_folder_path, v3dswc_folder_path)

    # copy v3dswc dir
    shutil.copytree(v3dswc_folder_path, v3dswc_copy_folder_path)
    get_list_traced(v3dswc_folder_path, list_traced_path)
    list_10847 = r"E:\tracing_ws\10847\list_10847.xlsx"
    swc_list = [f for f in os.listdir(v3dswc_copy_folder_path) if f.endswith('.swc')]
    df_list = pd.read_excel(list_10847)['ID'].values
    for swc_file in swc_list:
        ID = int(swc_file.split('/')[-1].split('_')[0])
        if (ID not in df_list):
            print(ID)
            if (os.path.exists(os.path.join(v3dswc_copy_folder_path, swc_file))):
                os.remove(os.path.join(v3dswc_copy_folder_path, swc_file))
            continue

    pass


def rename_muti_soma_markers(muti_soma_marker_folder):
    file_names = os.listdir(muti_soma_marker_folder)
    for file_name in file_names:
        if ("marker" in file_name):
            os.rename(os.path.join(muti_soma_marker_folder, file_name),
                      os.path.join(muti_soma_marker_folder, file_name.replace('v3draw.marker', 'marker')))


if __name__ == '__main__':
    # temp_prepossessing()
    # rename_muti_soma_markers(muti_soma_marker_folder_path)

    prepossessing()
    # folder1 = r"E:\tracing_ws\10847\TEST10K7\tif"
    # folder2 = r"E:\tracing_ws\10847\TEST10K1\tif"
    # out_folder = r"E:\tracing_ws\10847\TEST10K7\compare"
    # compare_tif(folder1, folder2, out_folder)
    tracing()
    postprocessing()

    # num = 0
    # file_names = os.listdir(muti_soma_marker_folder_path)
    # for file_name in file_names:
    #     v3dswc_path = os.path.join(v3dswc_folder_path, os.path.splitext(file_name)[0] + '.swc')
    #     if(v3dswc_path and os.path.exists(v3dswc_path)):
    #         num += 1
    #         # copy
    #         shutil.copy(os.path.join(muti_soma_marker_folder_path, file_name), os.path.join(muti_soma_marker_folder_path, file_name.replace('v3draw.marker', 'marker')))
    # print(num)
