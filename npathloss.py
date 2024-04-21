import time

import cc3d
import numpy as np
import tifffile
import torch
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation

from nnunetv2.training.loss.fmm_preprocess2 import find_path_to_source, mip_and_path_visualization
import os
from nnunetv2.training.loss.fmm.fmm_path import calculate_path_loss_mask


def soma_cc_ndimage(image, source):
    """
    Finds the connected component containing the source point in a 3D image.

    Parameters:
    - image: A numpy array representing the 3D binary image (foreground=1, background=0).
    - source: A tuple (x, y, z) representing the coordinates of the source point.

    Returns:
    - connected_component: A numpy array with the same shape as `image`, where the connected
      component containing the source point is marked as 1, and the rest is 0.
    """
    # Label all connected components in the image
    labeled_image, num_features = label(image)

    # Find the label of the connected component that contains the source point
    source_label = labeled_image[source]

    # Create a binary image marking the connected component containing the source point
    connected_component = (labeled_image == source_label).astype(int)

    return connected_component


def soma_cc_cc3d(image, source):
    """
    Finds the connected component containing the source point in a 3D image.

    Parameters:
    - image: A numpy array representing the 3D binary image (foreground=1, background=0).
    - source: A tuple (x, y, z) representing the coordinates of the source point.

    Returns:
    - connected_component: A numpy array with the same shape as `image`, where the connected
      component containing the source point is marked as 1, and the rest is 0.
    """
    # Label all connected components in the image
    labeled_image = cc3d.connected_components(image, connectivity=26)

    # Find the label of the connected component that contains the source point
    source_label = labeled_image[source]

    # Create a binary image marking the connected component containing the source point
    connected_component = (labeled_image == source_label).astype(int)

    return connected_component


def random_foreground_points(image, m):
    """
    从二值化图像中随机选择m个前景点。

    Parameters:
    - image: 二值化图像，numpy数组，前景点像素值为1。
    - m: 需要随机选择的前景点的数量。

    Returns:
    - points: 随机选择的前景点的坐标列表，每个坐标为(z, y, x)形式的元组（对于3D图像）或(y, x)形式的元组（对于2D图像）。
    """
    # 找出所有前景点的坐标
    foreground_coords = np.argwhere(image == 1)

    # 检查是否有足够的前景点
    if len(foreground_coords) < m:
        # raise ValueError("The image does not have enough foreground points.")
        m = len(foreground_coords)

    # 从前景点坐标中随机选择m个
    chosen_indices = np.random.choice(len(foreground_coords), m, replace=False)
    chosen_coords = foreground_coords[chosen_indices]

    # 将numpy数组转换为坐标列表
    points = [tuple(coord) for coord in chosen_coords]

    return points


def find_foreground_points(binary_image):
    """
    Finds the coordinates of the foreground points in a 3D binary image.

    Parameters:
    - binary_image: A 3D numpy array representing a binary image, where foreground points are denoted by 1.

    Returns:
    - A list of tuples, where each tuple represents the (x, y, z) coordinates of a foreground point.
    """
    # Ensure the input is a 3D numpy array
    if binary_image.ndim != 3:
        raise ValueError("Input image must be a 3D numpy array.")

    # Find indices where the value is 1 (foreground points)
    foreground_coords = np.argwhere(binary_image == 1)

    # Convert indices to list of tuples
    foreground_points = [tuple(coord) for coord in foreground_coords]

    return foreground_points

def npathloss(gt, pred, seg, predecessor, num_paths=10, soma=None, debug=False, threshold=0.5, smooth=1e-6):
    if(soma is None or predecessor is None):
        return 0

    # print(num_paths)
    gt_clone = gt.detach().clone().cpu().numpy()
    pred_clone = pred.detach().clone().cpu().numpy()
    predecessor_clone = predecessor.detach().clone().cpu().numpy()
    soma_clone = soma.detach().clone().cpu().numpy()
    soma_clone = tuple([int(soma_clone[0]), int(soma_clone[1]), int(soma_clone[2])])
    if(soma_clone[0] + soma_clone[1] + soma_clone[2] == 0):
        # print('fuck0')
        return 0
    # print(f"max {max(seg.flatten())}, min {min(seg.flatten())}")
    if(sum(predecessor_clone.flatten()) == -1 * len(predecessor_clone.flatten())):
        # print('fuck0')
        return 0
    if(sum(seg.flatten()) / (seg.shape[0] * seg.shape[1] * seg.shape[2]) > 0.05):
        # print("fuck0")
        return 0
    bin_pred = seg
    # bin_pred = binary_erosion(bin_pred, iterations=3)
    soma_cc = soma_cc_cc3d(bin_pred, soma_clone)
    soma_cc = binary_dilation(soma_cc, iterations=3)

    non_soma_cc = gt_clone - soma_cc
    non_soma_cc = np.where(non_soma_cc > 0.5, 1, 0).astype(int)

    rand_points = random_foreground_points(non_soma_cc, num_paths)
    pt_loss = 0

    if (debug):
        paths = []
    for start_point in rand_points:
        mask_path, mask_path_from_soma = calculate_path_loss_mask(predecessor_clone, bin_pred, start_point, soma_clone, threshold)
        if(debug):
            # path = find_foreground_points(mask_path)
            path = find_path_to_source(predecessor_clone, start_point, soma_clone)
            paths.append(path)
        mask_path = torch.from_numpy(mask_path).to(pred.device)
        mask_path_from_soma = torch.from_numpy(mask_path_from_soma).to(pred.device)

        ptls1 = (threshold - pred) * mask_path
        ptls1 = torch.relu(ptls1)
        ptls2 = (1 - pred) * mask_path_from_soma

        # print(f"torch.sum(ptls1) * torch.sum(ptls2) {torch.sum(ptls1)} {torch.sum(ptls2)}")

        pt_loss = pt_loss + (torch.sum(ptls1) + smooth) * (torch.sum(ptls2) + smooth) / (mask_path.sum() ** 2 * 0.5)


    if (debug):
        dir_path = "/home/kfchen/temp_mip/"
        file_numbers = len([lists for lists in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, lists))])
        if (file_numbers > 60):
            pass
            os.remove(dir_path + os.listdir(dir_path)[-1])
        else:
            temp_file_path = "/home/kfchen/temp_mip/" + str(time.time()) + "_" + str(pt_loss)
            if(len(rand_points) == 0):
                temp_num_paths = 0
            else:
                temp_num_paths = 10
            mip_and_path_visualization(seg.astype(np.uint8), paths, temp_file_path+"seg.png", temp_num_paths)
            # mip_and_path_visualization(soma_cc, paths, temp_file_path+"soma_cc.png", temp_num_paths)
            # mip_and_path_visualization(non_soma_cc, paths, temp_file_path+"non_soma_cc.png", temp_num_paths)
            mip_and_path_visualization(gt_clone, paths, temp_file_path+"gt.png", temp_num_paths)
            # print(gt_clone.shape)
            # print(gt_clone.dtype)
            # print(max(gt_clone.flatten()), min(gt_clone.flatten()))
            # gt_clone = np.where(gt_clone > 0.5, 1, 0).astype(np.uint8) * 255
            # print(gt_clone.shape)
            # print(gt_clone.dtype)
            # print(max(gt_clone.flatten()), min(gt_clone.flatten()))
            # tifffile.imwrite(temp_file_path+"gt.tif", gt_clone)
            # tifffile.imwrite(temp_file_path+"seg.tif", seg.cpu().numpy().astype(np.uint8) * 255)




    # print(f"pt_loss {pt_loss}, rand_points {len(rand_points)}")
    return pt_loss
