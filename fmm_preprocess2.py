import numpy as np
import heapq
import os
import tifffile
from fmm_preprocess import get_soma
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

def encode_neighbor(dx, dy, dz):
    """
    Encode the neighbor offset (dx, dy, dz) into a single integer.
    """
    return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)

def decode_neighbor(code):
    """
    Decode the single integer code back into the neighbor offset (dx, dy, dz).
    """
    # code -= 1
    dz = code % 3 - 1
    code //= 3
    dy = code % 3 - 1
    dx = code // 3 - 1
    return dx, dy, dz


def compute_fast_marching(image, source):
    """
    Computes the shortest distances from a source point to all foreground points
    in a 3D image using a breadth-first search with a priority queue (min heap).

    Parameters:
    - image: A numpy array representing the 3D binary image (foreground=1, background=0).
    - source: A tuple (x, y, z) representing the coordinates of the source point.

    Returns:
    - distance_image: A numpy array of the same shape as `image`, where each element
      is the shortest distance from the source to that point.
    """
    # Initialize the distance image with infinity values
    distance_image = np.full(image.shape, np.inf)
    predecessor_image = np.full(image.shape, -1, dtype=int)
    source = tuple(map(int, source))
    # Set the distance to the source point to 0
    distance_image[source] = 0

    # Define the 26-neighbourhood for a 3D grid
    neighbors_offsets = [(i, j, k) for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1] if (i, j, k) != (0, 0, 0)]
    neighbors_codes = {offset: encode_neighbor(*offset) for offset in neighbors_offsets}

    # Use a min heap as the priority queue
    # Each element in the heap is a tuple (distance, (x, y, z))
    heap = [(0, source)]

    while heap:
        current_distance, (x, y, z) = heapq.heappop(heap)

        # Explore all neighbours
        for dx, dy, dz in neighbors_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            # Check if the neighbour is within the image bounds
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and 0 <= nz < image.shape[2]:
                # Calculate the distance to the neighbour (Euclidean distance for simplicity)
                distance_to_neighbour = current_distance + np.sqrt(dx**2 + dy**2 + dz**2)
                # If the new calculated distance is less than the current distance at the neighbour,
                # update the distance and add the neighbour to the heap
                if image[nx, ny, nz] == 1 and distance_to_neighbour < distance_image[nx, ny, nz]:
                    distance_image[nx, ny, nz] = distance_to_neighbour
                    # Encode the neighbor's connection as an integer
                    predecessor_image[nx, ny, nz] = neighbors_codes[(dx, dy, dz)]
                    heapq.heappush(heap, (distance_to_neighbour, (nx, ny, nz)))

    distance_image = np.where(distance_image == np.inf, 0, distance_image)

    return distance_image, predecessor_image


def get_predecessor_info(predecessor_image):
    valid_points = np.argwhere(predecessor_image != -1)
    valid_values = predecessor_image[predecessor_image != -1]
    return valid_points, valid_values
def save_predecessor_info(valid_points, valid_values, filename):
    np.savez(filename, points=valid_points, values=valid_values)

def load_predecessor_info(filename):
    data = np.load(filename)
    points, values = data['points'], data['values']
    # 你需要根据这些点和值重建predecessor_image，或者直接使用这些信息
    return points, values


def find_path_to_source(predecessor_image):
    # 找到所有值不为-1的点的坐标
    valid_points = np.argwhere(predecessor_image != -1)

    # 如果没有找到有效点，返回空路径
    if len(valid_points) == 0:
        return []

    # 随机选择一个有效点作为起点
    start_point_idx = np.random.choice(len(valid_points))
    start_point = valid_points[start_point_idx]

    # 初始化路径，起始点加入路径
    path = [tuple(start_point)]

    # 从起点开始追溯前驱点，直到找到源点
    current_point = start_point
    while True:
        # 获取当前点的前驱点编码
        pred_code = predecessor_image[tuple(current_point)]

        # 如果到达源点（编码为-1），则终止追溯
        if pred_code == -1:
            break

        # 解码前驱点的坐标偏移
        dx, dy, dz = decode_neighbor(pred_code)

        # 计算前驱点坐标
        current_point = current_point + np.array([dx, dy, dz])

        # 将前驱点加入路径
        path.append(tuple(current_point))

    return path

def find_random_paths(predecessor_image, num_paths=5):
    paths = []  # 用于存储所有路径
    # 找到所有值不为-1的点的坐标
    valid_points = np.argwhere(predecessor_image != -1)

    # 如果没有找到有效点，或者请求的路径数多于有效点的数量，返回空列表
    if len(valid_points) == 0 or num_paths > len(valid_points):
        return []

    # 随机选择m个不重复的索引
    start_point_idxs = np.random.choice(len(valid_points), size=num_paths, replace=False)

    for idx in start_point_idxs:
        path = find_path_to_source(predecessor_image, idx)
        paths.append(path)  # 将找到的路径加入到paths列表中
    return paths
def find_path_to_source(predecessor_image, idx):
    start_point = valid_points[idx]  # 获取起点坐标
    path = [tuple(start_point)]  # 初始化路径，起始点加入路径

    current_point = start_point
    while True:
        pred_code = predecessor_image[tuple(current_point)]
        if pred_code == -1:  # 到达源点
            break
        # 解码前驱点的坐标偏移
        dx, dy, dz = decode_neighbor(pred_code)
        # 计算前驱点坐标
        current_point = current_point - np.array([dx, dy, dz])
        # 将前驱点加入路径
        path.append(tuple(current_point))
    # print(path)
    return path


def rebuild_predecessor_image(shape, points, values):
    # 初始化predecessor_image，所有元素初始化为-1
    predecessor_image = np.full(shape, -1, dtype=np.int32)

    # 根据valid_points和valid_values填充predecessor_image
    for point, value in zip(points, values):
        predecessor_image[tuple(point)] = value
    return predecessor_image


def mip_and_path_visualization(image, paths, temp_mip_path, num_paths=5):
    """
    在XY平面上计算三维图像的MIP，并在MIP上绘制从源点到目标点的路径。
    """
    # 计算XY平面的MIP
    mip_xy = np.max(image, axis=0)

    # 绘制MIP和路径
    plt.figure(figsize=(10, 8))
    plt.imshow(mip_xy, cmap='gray')
    plt.colorbar(label='Distance to source point')
    selected_paths = random.sample(paths, min(num_paths, len(paths)))  # 随机选取num_paths条路径

    for i, path in enumerate(selected_paths):
        path = np.array(path)
        # 将路径转换为XY坐标并绘制
        path_xy = path[:, [1, 2]]  # 提取Y和X坐标
        plt.plot(path_xy[:, 1], path_xy[:, 0], label=f'Path {i+1}', linewidth=2)  # 绘制路径

    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('MIP on XY Plane with Path')
    plt.savefig(temp_mip_path)
    plt.close()

def get_fmm(tif_path, fmm_folder, path_folder, temp_path, debug=False):
    distance_image_path = os.path.join(fmm_folder, os.path.basename(tif_path))
    predecessor_image_path = os.path.join(path_folder, os.path.basename(tif_path).replace('.tif', '.npz'))
    if(os.path.exists(predecessor_image_path)):
        return
    img = tifffile.imread(tif_path)
    source = get_soma(img, tif_path, temp_path=temp_path)
    distance_image, predecessor_image = compute_fast_marching(img, source)
    # distance_image = np.where(distance_image, 1/distance_image, 0)
    # Normalize the distance image to [0, 255] and save it as a uint8 image
    if (debug):
        distance_image = (distance_image - distance_image.min()) / (distance_image.max() - distance_image.min())
        distance_image = (distance_image * 255).astype(np.uint8)
        tifffile.imwrite(distance_image_path, distance_image)
    valid_points, valid_values = get_predecessor_info(predecessor_image)
    save_predecessor_info(valid_points, valid_values, predecessor_image_path)


if __name__ == '__main__':
    debug = False

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


    if(debug):
        tif_list = tif_list[:1]

    partial_func = partial(get_fmm, fmm_folder=fmm_folder, path_folder=path_folder, temp_path=temp_path, debug=debug)
    with Pool(5) as p:
        for _ in tqdm(p.imap_unordered(partial_func, tif_list), total=len(tif_list)):
            pass

    # for tif_path in tif_list:
    #     distance_image_path = os.path.join(fmm_folder, os.path.basename(tif_path))
    #     predecessor_image_path = os.path.join(path_folder, os.path.basename(tif_path).replace('.tif', '.npz'))
    #     img = tifffile.imread(tif_path)
    #     source = get_soma(img, tif_path, temp_path=temp_path)
    #     distance_image, predecessor_image = compute_fast_marching(img, source)
    #     # distance_image = np.where(distance_image, 1/distance_image, 0)
    #     # Normalize the distance image to [0, 255] and save it as a uint8 image
    #     if(debug):
    #         distance_image = (distance_image - distance_image.min()) / (distance_image.max() - distance_image.min())
    #         distance_image = (distance_image*255).astype(np.uint8)
    #         tifffile.imwrite(distance_image_path, distance_image)
    #     valid_points, valid_values = get_predecessor_info(predecessor_image)
    #     save_predecessor_info(valid_points, valid_values, predecessor_image_path)

    if(debug):
        for tif_path in tif_list:
            distance_image_path = os.path.join(fmm_folder, os.path.basename(tif_path))
            predecessor_image_path = os.path.join(path_folder, os.path.basename(tif_path).replace('.tif', '.npz'))

            distance_image = tifffile.imread(distance_image_path)
            valid_points, valid_values = load_predecessor_info(predecessor_image_path)
            predecessor_image = rebuild_predecessor_image(tifffile.imread(tif_path).shape, valid_points, valid_values)
            # path = find_path_to_source(predecessor_image)
            # print(path)
            # print("find path")
            m_paths = find_random_paths(predecessor_image, num_paths=5)
            mip_path = os.path.join(temp_mip_folder, os.path.basename(tif_path).replace('.tif', '.png'))
            mip_and_path_visualization(distance_image, m_paths, mip_path, num_paths=5)








