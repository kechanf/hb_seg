# cython: language_level=3
import numpy as np
cimport numpy as cnp
import heapq
from libc.math cimport sqrt
import torch

def encode_neighbor(int dx, int dy, int dz):
    """
    A simple function to encode neighbor offsets.
    """
    return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)

cpdef tuple decode_neighbor(int code):
    """
    Decode the single integer code back into the neighbor offset (dx, dy, dz).
    """
    cdef int dx, dy, dz
    dz = code % 3 - 1
    code //= 3
    dy = code % 3 - 1
    dx = code // 3 - 1
    return dx, dy, dz

def compute_fast_marching(cnp.ndarray image, tuple source, tuple shape):
    cdef:
        cnp.ndarray distance_image
        cnp.ndarray predecessor_image
        int x, y, z, nx, ny, nz, dx, dy, dz
        float distance_to_neighbour, current_distance
        list heap = []
        tuple node

    # Manually construct the shape tuple

    # Initialize the distance image with infinity values
    distance_image = np.full(shape, np.inf)
    predecessor_image = np.full(shape, -1)

    # Convert source to int tuple in case it's not
    source = tuple(map(int, source))
    distance_image[source] = 0.0

    # Define the 26-neighbourhood for a 3D grid
    neighbors_offsets = [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2) if
                         (i, j, k) != (0, 0, 0)]
    # Construct neighbors_codes dictionary in Python space
    neighbors_codes = {offset: encode_neighbor(*offset) for offset in neighbors_offsets}

    heapq.heappush(heap, (0.0, source))

    while heap:
        current_distance, node = heapq.heappop(heap)
        x, y, z = node

        for dx, dy, dz in neighbors_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz

            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                distance_to_neighbour = current_distance + sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                if image[nx, ny, nz] == 1 and distance_to_neighbour < distance_image[nx, ny, nz]:
                    distance_image[nx, ny, nz] = distance_to_neighbour
                    predecessor_image[nx, ny, nz] = neighbors_codes[(dx, dy, dz)]
                    heapq.heappush(heap, (distance_to_neighbour, (nx, ny, nz)))

    distance_image = np.where(distance_image == np.inf, 0, distance_image)

    return distance_image, predecessor_image

cpdef list find_path_to_source(cnp.ndarray predecessor_image, tuple start_point, tuple source=None):
    cdef list path = [start_point]  # 初始化路径，起始点加入路径
    cdef cnp.ndarray current_point = np.array(start_point, dtype=np.int32)  # 确保为整数类型数组
    cdef int pred_code
    cdef (int, int, int) offsets

    while True:
        pred_code = predecessor_image[tuple(current_point)]
        if source is not None and tuple(current_point) == source:
            break
        if pred_code == -1:  # 到达源点
            break
        # 解码前驱点的坐标偏移，这里需要提供decode_neighbor的Cython版本或者将其直接包含在这里
        offsets = decode_neighbor(pred_code)
        # 计算前驱点坐标
        current_point -= np.array([offsets[0], offsets[1], offsets[2]], dtype=np.int32)

        # 将前驱点加入路径
        path.append(tuple(current_point))
        if(len(path) > 400):
            break

    return path

def calculate_path_loss_mask(predecessor_clone, bin_pred, start_point, soma_clone, threshold):

    path = find_path_to_source(predecessor_clone, start_point, soma_clone)
    path = path[::-1]

    mask_path = np.zeros_like(predecessor_clone, dtype=np.uint8)
    mask_path_from_soma = np.zeros_like(predecessor_clone, dtype=np.uint8)

    continue_from_soma_flag = True
    for point in path:
        mask_path[point] = 1
        if bin_pred[point] < threshold:
            continue_from_soma_flag = False
        if not continue_from_soma_flag:
            mask_path_from_soma[point] = 1

    return mask_path, mask_path_from_soma



