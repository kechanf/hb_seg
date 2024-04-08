from nnunetv2.training.loss.fmm_preprocess2 import find_path_to_source, mip_and_path_visualization
import numpy as np
import cc3d
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import label
import time
import tifffile
import torch

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
    labeled_image = cc3d.connected_components(image, connectivity=6)

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

def npathloss(gt, pred, predecessor, num_paths=10, soma=None, debug=False, threshold=0.5):
    gt_clone = gt.detach().clone().cpu().numpy()
    pred_clone = pred.detach().clone().cpu().numpy()
    predecessor_clone = predecessor.detach().clone().cpu().numpy()
    soma_clone = soma.detach().clone().cpu().numpy()
    soma_clone = tuple([int(soma_clone[0]), int(soma_clone[1]), int(soma_clone[2])])

    bin_pred = (pred_clone > threshold).astype(int)
    # bin_pred = binary_erosion(bin_pred, iterations=3)
    soma_cc = soma_cc_cc3d(bin_pred, soma_clone)
    soma_cc = binary_dilation(soma_cc, iterations=3)

    non_soma_cc = gt_clone - soma_cc
    non_soma_cc = np.where(non_soma_cc > 0.5, 1, 0).astype(int)

    rand_points = random_foreground_points(non_soma_cc, num_paths)
    pt_loss = 0
    if(debug):
        paths = []
    for start_point in rand_points:
        # print(f"point: {point}, {point[0]}")
        # print(f"type of point: {type(point)}, {type(point[0])}")
        path = find_path_to_source(predecessor_clone, start_point, soma_clone)
        if(debug):
            paths.append(path)
        # 反转path， 即从soma开始
        path = path[::-1]
        # print(f'start_point: {start_point}, path[0]: {path[0]}, path[-1]: {path[-1]}, {bin_pred[path[0]]}, {bin_pred[path[-1]]}')
        mask_path = np.zeros_like(gt_clone)
        mask_path_from_soma = np.zeros_like(gt_clone)
        continue_from_soma_flag = True
        for point in path:
            mask_path[point] = 1
            if(bin_pred[point] < threshold): # 第一个中断点
                continue_from_soma_flag = False
            if(not continue_from_soma_flag):
                mask_path_from_soma[point] = 1
        # to tensor, and change to device
        mask_path = torch.from_numpy(mask_path).to(pred.device)
        mask_path_from_soma = torch.from_numpy(mask_path_from_soma).to(pred.device)

        ptls1 = (threshold - pred) * mask_path
        ptls1 = torch.relu(ptls1)
        ptls2 = (1 - pred) * mask_path_from_soma

        # print(f"torch.sum(ptls1) * torch.sum(ptls2) {torch.sum(ptls1)} {torch.sum(ptls2)}")

        pt_loss = pt_loss + torch.sum(ptls1) * torch.sum(ptls2)


    if(debug):
        temp_file_path = "/home/kfchen/temp_mip/" + str(time.time()) + ".png"
        mip_and_path_visualization(pred_clone, paths, temp_file_path, num_paths)


    return pt_loss



    














