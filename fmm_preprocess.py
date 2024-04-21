import time

import numpy as np
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
import tifffile
import torch

from scipy.ndimage import binary_dilation
import os
import subprocess
import sys
import cc3d
import cupy as cp
from cupyx.scipy.ndimage import binary_opening
import cupyx
from skimage.morphology import ball
import scipy
import skfmm

import numpy as np
# from scipy.ndimage import gradient
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
from tqdm import tqdm
import random
import pickle
from multiprocessing import Pool
from functools import partial

def dusting(img):
    if(img.sum() == 0):
        return img
    labeled_image = cc3d.connected_components(img, connectivity=6)
    largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    largest_component_binary = ((labeled_image == largest_label)).astype(np.uint8)
    return largest_component_binary

def process_path(pstr):
    return pstr.replace('(', '\(').replace(')', '\)')

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
        spherical_selem = ball(radius * (max_rate-i)/10 / 2)
        # soma_region_res = binary_opening(soma_region, spherical_selem).astype(np.uint8)
        soma_region_res = scipy.ndimage.binary_opening(soma_region, spherical_selem).astype(np.uint8)
        if(soma_region_res.sum() == 0):
            continue
        soma_region = soma_region_res

    # soma_region = binary_erosion(soma_region, spherical_selem).astype(np.uint8)
    del spherical_selem, radius, soma_region_res
    if (soma_region.sum() == 0):
        soma_region = soma_region_copy
    del soma_region_copy

    return soma_region


def get_min_diameter_3d(binary_image):
    labeled_array, num_features = scipy.ndimage.label(binary_image)
    largest_cc = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    slice_x, slice_y, slice_z = scipy.ndimage.find_objects(labeled_array == largest_cc)[0]
    diameter_x = slice_x.stop - slice_x.start
    diameter_y = slice_y.stop - slice_y.start
    diameter_z = slice_z.stop - slice_z.start

    return min(diameter_x, diameter_y, diameter_z)

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

import skimage
# def compute_centroid(mask):
#     # 计算三维 mask 的重心
#     labeled_mask = skimage.measure.label(mask)
#     # props = regionprops(labeled_mask)
#     props = skimage.measure.regionprops(labeled_mask)
#
#     if len(props) > 0:
#         # 获取第一个区域的重心坐标
#         centroid = props[0].centroid
#         return centroid
#     else:
#         return None

def compute_centroid(mask):
    # 使用 cc3d 对3D mask进行连通区域标记
    labels = cc3d.connected_components(mask)  # 默认情况下, cc3d 会返回一个与 mask 同形状的标记数组

    # 初始化最大连通块的体积为0和其标签
    max_volume = 0
    max_label = 0

    # 找出最大的连通块
    for label in np.unique(labels):
        if label == 0:  # 跳过背景
            continue
        volume = (labels == label).sum()
        if volume > max_volume:
            max_volume = volume
            max_label = label

    # 如果没有找到连通块，则返回None
    if max_label == 0:
        return None

    # 计算最大连通块的重心
    coords = np.argwhere(labels == max_label)
    centroid = coords.mean(axis=0)

    return tuple(centroid)

def get_soma(img, img_path, temp_path=r"/home/kfchen/temp_tif", v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    # temp_path=r"C:\Users\12626\Desktop\topo_test\temp_path
    # v3d_path=r"D:\Vaa3D-x.1.1.4_Windows_64bit_version\Vaa3D-x.exe"
    file_name = os.path.basename(img_path)
    # print(file_name)
    in_tmp = os.path.join(temp_path, file_name+'temp.tif')
    tifffile.imwrite(in_tmp, img*255)
    out_tmp = in_tmp.replace('.tif', '_gsdt.tif')

    if(sys.platform == "linux"):
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x gsdt -f gsdt -i {in_tmp} -o {out_tmp} -p 0 1 0 1.5'
        cmd_str = process_path(cmd_str)
        # print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
    else:
        cmd = f'{v3d_path} /x gsdt /f gsdt /i {in_tmp} /o {out_tmp} /p 0 1 0 1.5'
        # print(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    pred = tifffile.imread(in_tmp).astype(np.uint8)
    pred[pred <= 255 / 2] = 0
    pred[pred > 255 / 2] = 1

    try:
        gsdt = tifffile.imread(out_tmp).astype(np.uint8)
        gsdt = np.flip(gsdt, axis=1)
        if (os.path.exists(out_tmp)): os.remove(out_tmp)
        if(os.path.exists(in_tmp)): os.remove(in_tmp)
        del out_tmp, in_tmp
    except:
        print(f"error in {in_tmp}")
        print(cmd_str)
        return None

    # save tif
    # tifffile.imwrite(r"E:\tracing_ws\10847\TEST10K5\tif\111.tif", gsdt)


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

    return compute_centroid(soma_region)

def simple_get_soma(img, img_path, temp_path=r"/home/kfchen/temp_tif", v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    file_name = os.path.basename(img_path)
    # print(file_name)
    in_tmp = os.path.join(temp_path, file_name + 'temp.tif')
    tifffile.imwrite(in_tmp, (img * 255).astype(np.uint8))
    out_tmp = in_tmp.replace('.tif', '_gsdt.tif')

    if (sys.platform == "linux"):
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x gsdt -f gsdt -i {in_tmp} -o {out_tmp} -p 0 1 0 1.5'
        cmd_str = process_path(cmd_str)
        # print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
    else:
        cmd = f'{v3d_path} /x gsdt /f gsdt /i {in_tmp} /o {out_tmp} /p 0 1 0 1.5'
        # print(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if(not os.path.exists(out_tmp)):
        return None
    gsdt = tifffile.imread(out_tmp).astype(np.uint8)
    gsdt = np.flip(gsdt, axis=1)

    max_gsdt = np.max(gsdt)
    gsdt[gsdt <= max_gsdt * 0.95] = 0
    gsdt[gsdt > max_gsdt * 0.95] = 1

    centroid = compute_centroid(gsdt)

    if (os.path.exists(out_tmp)): os.remove(out_tmp)
    if (os.path.exists(in_tmp)): os.remove(in_tmp)
    del out_tmp, in_tmp

    return centroid




def compute_fmm_single_source(binary_image, source_point):
    """
    对二值图像进行距离变换预处理，并针对单一源点计算Fast Marching Method (FMM)。

    参数:
    - binary_image: 二值化的三维图像（numpy数组），其中前景为1，背景为0。
    - source_point: 单一源点的坐标，格式为(z, y, x)。

    返回值:
    - fmm_distance_field: 计算得到的FMM距离场。
    """
    # 创建一个与二值图像相同形状的数组，所有值均为无穷大
    phi = np.ones_like(binary_image, dtype=np.float32) * np.inf

    # 对背景执行距离变换，获取每个背景像素到最近前景像素的距离，作为成本矩阵
    distance_transformed = distance_transform_edt(binary_image == 0)

    # 将背景的距离变换结果作为成本矩阵的背景值，使FMM可以基于这个成本推进
    phi[binary_image == 0] = distance_transformed[binary_image == 0]

    # 将源点周围的区域（这里设为源点自身）设置为0，表示成本最低
    phi[source_point] = 0
    binary_image[source_point] = 1

    # print(source_point)
    # print(phi[source_point])
    # print(binary_image[source_point])

    # 使用skfmm进行FMM计算
    fmm_distance_field = skfmm.distance(phi, dx=1.0)
    distance_field_foreground = np.where(binary_image, fmm_distance_field, 0)

    return distance_field_foreground


def trace_path_to_source(fmm_distance_field, source_point, target_point):
    """
    追踪从目标点到源点的路径。
    """
    # 计算梯度场
    gradz, grady, gradx = np.gradient(-fmm_distance_field)

    # 创建插值函数
    interpolator_x = RegularGridInterpolator((np.arange(fmm_distance_field.shape[0]),
                                              np.arange(fmm_distance_field.shape[1]),
                                              np.arange(fmm_distance_field.shape[2])), gradx)
    interpolator_y = RegularGridInterpolator((np.arange(fmm_distance_field.shape[0]),
                                              np.arange(fmm_distance_field.shape[1]),
                                              np.arange(fmm_distance_field.shape[2])), grady)
    interpolator_z = RegularGridInterpolator((np.arange(fmm_distance_field.shape[0]),
                                              np.arange(fmm_distance_field.shape[1]),
                                              np.arange(fmm_distance_field.shape[2])), gradz)

    path = [target_point]
    current_point = np.array(target_point, dtype=np.float64)

    # 梯度下降迭代
    while np.linalg.norm(current_point - source_point) > 1:
        grad = np.array([interpolator_z(current_point),
                         interpolator_y(current_point),
                         interpolator_x(current_point)])
        norm_grad = np.linalg.norm(grad)
        if norm_grad == 0:  # 防止除以0的错误
            break

        # 更新current_point，确保广播操作不会引发错误
        current_point[0] -= grad[0] / norm_grad * 0.5
        current_point[1] -= grad[1] / norm_grad * 0.5
        current_point[2] -= grad[2] / norm_grad * 0.5

        path.append(current_point.copy())

    return path


def naive_trace_path(fmm_distance_field, source_point, foreground_mask, neighbor_directions=1):
    """
       使用26邻域、优先队列和FMM距离信息进行启发式搜索，从源点到所有前景点的路径，显示进度。
       """
    # 初始化优先队列，包含源点和初始路径
    priority_queue = [(0, source_point, [source_point])]

    # 记录已访问的点
    visited = np.zeros_like(fmm_distance_field, dtype=bool)
    visited[source_point] = True

    # 存储路径信息
    paths = {}

    # 定义26个可能的邻居方向
    if(neighbor_directions == 0):
        neighbor_directions = [(x, y, z) for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1] if
                               (x, y, z) != (0, 0, 0)]
    # 6邻域
    elif(neighbor_directions == 1):
        neighbor_directions = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

    # 计算前景点的数量以显示进度
    total_foreground_points = np.sum(foreground_mask)
    # progress_bar = tqdm(total=total_foreground_points, desc="Processing foreground points")

    while priority_queue:
        # 按照FMM距离值弹出下一个点
        dist, current_point, current_path = heapq.heappop(priority_queue)

        # 如果当前点是前景点，记录路径并更新进度条
        # print(current_point)
        # print(foreground_mask[current_point])
        paths[current_point] = current_path
        # progress_bar.update(1)

        # 检查所有26个邻居
        for d in neighbor_directions:
            next_point = tuple(np.array(current_point) + np.array(d))
            if (np.any(np.array(next_point) < 0) or np.any(np.array(next_point) >= fmm_distance_field.shape)):
                continue
            if(visited[next_point]):
                continue
            visited[next_point] = True

            # 检查边界条件和是否已访问
            if np.any(np.array(next_point) < 0):
                continue
            if(foreground_mask[current_point] == 0):
                continue

            # 添加新点到优先队列
            heapq.heappush(priority_queue, (fmm_distance_field[next_point], next_point, current_path + [next_point]))


    # progress_bar.close()
    return paths


def mip_and_path_visualization(image, paths, source_point, temp_mip_path, num_paths=5):
    """
    在XY平面上计算三维图像的MIP，并在MIP上绘制从源点到目标点的路径。
    """

    # 计算XY平面的MIP
    mip_xy = np.max(image, axis=0)

    # 绘制MIP和路径
    plt.figure(figsize=(10, 8))
    plt.imshow(mip_xy, cmap='gray')
    plt.colorbar(label='Distance to source point')
    selected_keys = random.sample(list(paths.keys()), min(num_paths, len(paths)))

    for key in selected_keys:
        path = np.array(paths[key])
        # print(f"len of path: {len(path)}")
        # 将路径转换为XY坐标并绘制
        path_xy = np.array(path)[:, [1, 2]]  # 提取Y和X坐标
        plt.plot(path_xy[:, 1], path_xy[:, 0], label=f'Path to {key}', linewidth=2)  # 绘制路径

    # 标记源点和目标点
    plt.scatter(source_point[2], source_point[1], color='blue', s=100, label='Source')
    # plt.scatter(target_point[2], target_point[1], color='green', s=100, label='Target')

    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('MIP on XY Plane with Path')
    # plt.show()
    plt.savefig(temp_mip_path)

def save_fmm_distance_field(img, fmm_distance_field, file_path):
    fmm_distance_field = np.where(img, fmm_distance_field.max()-fmm_distance_field, 0)
    fmm_distance_field = (fmm_distance_field - fmm_distance_field.min()) / (fmm_distance_field.max() - fmm_distance_field.min()) * 255
    fmm_distance_field = fmm_distance_field.astype(np.uint8)
    tifffile.imwrite(file_path, fmm_distance_field)

def save_paths(paths, file_path):
    """
    将路径数据序列化并保存到文件中。

    参数:
    - paths: 包含所有路径的字典。
    - file_path: 要保存文件的路径。
    """
    with open(file_path, 'wb') as f:
        pickle.dump(paths, f)


def load_paths(file_path):
    """
    从文件中加载并反序列化路径数据。

    参数:
    - file_path: 包含路径数据的文件路径。

    返回:
    - 反序列化后的路径数据。
    """
    with open(file_path, 'rb') as f:
        paths = pickle.load(f)
    return paths


def get_fmm_path(tif_path, fmm_folder=r'C:\Users\12626\Desktop\topo_test', path_folder=r'C:\Users\12626\Desktop\topo_test', temp_path = r"/home/kfchen/temp_tif"):
    if(os.path.exists(fmm_folder) == False):
        os.makedirs(fmm_folder)
    if(os.path.exists(path_folder) == False):
        os.makedirs(path_folder)


    id = os.path.basename(tif_path).split('.')[0]

    fmm_path = os.path.join(fmm_folder, f'{id}.tif')
    path_path = os.path.join(path_folder, f'{id}.pkl')

    if(os.path.exists(fmm_path) and os.path.exists(path_path)):
        return
    if(os.path.exists(fmm_path)):
        os.remove(fmm_path)
    if(os.path.exists(path_path)):
        os.remove(path_path)

    img=tifffile.imread(tif_path)
    # binary to 0-1
    img = (img > 0.5).astype(np.uint8)

    source_point = get_soma(img, tif_path, temp_path=temp_path)
    if(source_point == None):
        return
    # to int
    source_point = tuple(map(int, source_point))

    # 计算Fast Marching Field
    fmm_distance_field = compute_fmm_single_source(img, source_point)
    save_fmm_distance_field(img, fmm_distance_field, fmm_path)

    # 追踪路径
    paths = naive_trace_path(fmm_distance_field, source_point, img)
    save_paths(paths, path_path)

def visualize_fmm_path(tif_path, fmm=None, fmm_folder=r'C:\Users\12626\Desktop\topo_test', paths=None, path_folder=r'C:\Users\12626\Desktop\topo_test',temp_mip_folder=r"", num_paths=5):
    # visualization
    if(paths==None):
        paths = load_paths(os.path.join(path_folder, os.path.basename(tif_path).split('.')[0]+'.pkl'))
    temp_mip_path = os.path.join(temp_mip_folder, os.path.basename(tif_path).split('.')[0]+'.png')
    if(fmm==None):
        fmm_distance_field = tifffile.imread(os.path.join(fmm_folder, os.path.basename(tif_path).split('.')[0]+'.tif'))
    # find source point have only one path
    source_points = [point for point, path in paths.items() if len(path) == 1]
    mip_and_path_visualization(fmm_distance_field, paths, source_points[0], temp_mip_path, num_paths)

def save_swc(tif_path, temp_swc_folder, num_paths=5):
    paths = load_paths(os.path.join(path_folder, os.path.basename(tif_path).split('.')[0] + '.pkl'))
    selected_keys = random.sample(list(paths.keys()), min(num_paths, len(paths)))
    p_num = 0
    swc_path = os.path.join(temp_swc_folder, os.path.basename(tif_path).split('.')[0] + '.swc')
    swc_str = ''
    soma_flag = False

    for i, key in enumerate(selected_keys):
        path = np.array(paths[key])
        for j, point in enumerate(np.array(path)):
            father = -1 if j == 0 else p_num
            p_num = p_num + 1
            if(soma_flag and j == 0):
                continue
            swc_str += f'{p_num} 247 {point[2]} {point[1]} {point[0]} 1 {father}\n'
            if(j == 0):
                soma_flag = True
    with open(swc_path, 'w') as f:
        f.write(swc_str)


if __name__ == '__main__':
    tif_folder = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset159_human_brain_10000_tpls/labelsTr'
    fmm_folder = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset159_human_brain_10000_tpls/fmm'
    path_folder = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset159_human_brain_10000_tpls/path'
    temp_swc_folder = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset159_human_brain_10000_tpls/temp_swc'
    temp_mip_folder = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset159_human_brain_10000_tpls/temp_mip"
    temp_path = r"/home/kfchen/temp_tif"
    for folder in [fmm_folder, path_folder, temp_swc_folder, temp_mip_folder]:
        if(os.path.exists(folder) == False):
            os.makedirs(folder)
    for file in os.listdir(temp_path):
        os.remove(os.path.join(temp_path, file))
    tif_list = os.listdir(tif_folder)
    tif_list = [os.path.join(tif_folder, tif) for tif in tif_list if tif.endswith('.tif')]

    debug = False
    if(debug):
        # delete all files in fmm_folder and path_folder
        # for file in os.listdir(fmm_folder):
        #     os.remove(os.path.join(fmm_folder, file))
        # for file in os.listdir(path_folder):
        #     os.remove(os.path.join(path_folder, file))
        tif_list = tif_list[:5]

    # partial_func = partial(get_fmm_path, fmm_folder=fmm_folder, path_folder=path_folder, temp_path=temp_path)
    # with Pool(5) as p:
    #     for _ in tqdm(p.imap_unordered(partial_func, tif_list), total=len(tif_list)):
    #         pass
    # for tif_path in tif_list:
    #     tif = tifffile.imread(tif_path)
    #     connected_components = cc3d.connected_components(tif, connectivity=26)
    #     num_components = np.max(connected_components)
    #     if(num_components > 1):
    #         print(f"num_components {num_components} in {tif_path}")

    if(debug):
        for tif_path in tif_list:
            visualize_fmm_path(tif_path, fmm_folder=fmm_folder, path_folder=path_folder, temp_mip_folder=temp_mip_folder, num_paths=10)
            save_swc(tif_path, temp_swc_folder, num_paths=10)





