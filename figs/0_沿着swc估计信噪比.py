import tifffile
import numpy as np
import os
import pandas as pd
from skimage.transform import resize
from joblib import Parallel, delayed
from tqdm import tqdm
from simple_swc_tool.swc_radius_estimator import SWC_Radius_Estimator
from scipy.ndimage import binary_dilation
from scipy.spatial import KDTree
from scipy.ndimage import label
import networkx as nx
import matplotlib.pyplot as plt

manual_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab"
mask_dir = "/data/kfchen/trace_ws/to_gu/mask"
rescaled_mask_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/seg_mask/1um_seg_mask"
manual_swc_with_radius_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_with_radius"
neuron_info_df = pd.read_csv("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv", encoding='gbk')
source_img_dir = "/data/kfchen/trace_ws/to_gu/img"
rescaled_img_dir = "/data/kfchen/trace_ws/to_gu/1um_img"
temp_result_save_dir = "/data/kfchen/trace_ws/signal_background_comp_test/temp_result_dir"


def rescale_mask_file(mask_file, rescaled_mask_file):
    if(os.path.exists(rescaled_mask_file)):
        return
    id = int(os.path.basename(mask_file).split('_')[0].split('.')[0])
    xy_resolution = neuron_info_df.loc[neuron_info_df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0]
    xy_resolution = float(xy_resolution) / 1000

    mask = tifffile.imread(mask_file)
    mask = mask.astype(np.float32)
    mask = np.where(mask > 0, 1, 0)
    mask = resize(mask, (mask.shape[0], int(mask.shape[1] * xy_resolution), int(mask.shape[2] * xy_resolution)), order=0, preserve_range=True)
    mask = mask.astype(np.uint8)
    tifffile.imwrite(rescaled_mask_file, mask)

def rescale_mask_dir(mask_dir, rescaled_mask_dir):
    os.makedirs(rescaled_mask_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    Parallel(n_jobs=10)(delayed(rescale_mask_file)(
        os.path.join(mask_dir, mask_file),
        os.path.join(rescaled_mask_dir, mask_file)) for mask_file in tqdm(mask_files))

def rescale_img_file(img_file, rescaled_img_file):
    if(os.path.exists(rescaled_img_file)):
        return
    id = int(os.path.basename(img_file).split('_')[0].split('.')[0])
    xy_resolution = neuron_info_df.loc[neuron_info_df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0]
    xy_resolution = float(xy_resolution) / 1000

    img = tifffile.imread(img_file)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = resize(img, (img.shape[0], int(img.shape[1] * xy_resolution), int(img.shape[2] * xy_resolution)), order=2, preserve_range=True)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype('uint8')
    tifffile.imwrite(rescaled_img_file, img)

def rescale_img_dir(source_img_dir, rescaled_img_dir):
    os.makedirs(rescaled_img_dir, exist_ok=True)
    img_files = [f for f in os.listdir(source_img_dir) if f.endswith('.tif')]
    Parallel(n_jobs=20)(delayed(rescale_img_file)(
        os.path.join(source_img_dir, img_file),
        os.path.join(rescaled_img_dir, img_file)) for img_file in tqdm(img_files))

def estimate_radius_file(swc_file, mask_file, swc_with_radius_file):
    if(os.path.exists(swc_with_radius_file)):
        return
    _ = SWC_Radius_Estimator(swc_file, mask_file, background_tolerance_ratio=0.1, flip_y=False, output_swc_file=swc_with_radius_file)

def estimate_radius_dir(manual_swc_dir, rescaled_mask_dir, manual_swc_with_radius_dir):
    os.makedirs(manual_swc_with_radius_dir, exist_ok=True)
    swc_files = [f for f in os.listdir(manual_swc_dir) if f.endswith('.swc')]
    # swc_files = sorted(swc_files, key=lambda x: int(x.split('.')6[0]))
    Parallel(n_jobs=8)(delayed(estimate_radius_file)(
        os.path.join(manual_swc_dir, swc_file),
        os.path.join(rescaled_mask_dir, swc_file.replace('.swc', '.tif')),
        os.path.join(manual_swc_with_radius_dir, swc_file)) for swc_file in tqdm(swc_files))
    # for swc_file in tqdm(swc_files):
    #     estimate_radius_file(
    #         os.path.join(manual_swc_dir, swc_file),
    #         os.path.join(rescaled_mask_dir, swc_file.replace('.swc', '.tif')),
    #         os.path.join(manual_swc_with_radius_dir, swc_file))

def get_surrounding_mask(mask, iterations=2):
    # dilate the mask
    mask = np.where(mask > 0, 1, 0)
    mask = mask.astype(np.uint8)
    dilate_mask = binary_dilation(mask, iterations=iterations)
    surrounding_mask = dilate_mask - mask
    surrounding_mask = np.where(surrounding_mask > 0, 1, 0)
    return surrounding_mask


def compute_mean_intensity_list(img, mask, sampling_points):
    """
        计算每个采样点的所有关联前景体素的平均体素强度，使用KD树加速最近邻查询。

        :param img: 输入的三维图像（numpy数组），包含体素强度。
        :param mask: 二值前景掩膜（numpy数组），表示前景体素位置。
        :param sampling_points: 采样点坐标列表，每个元素是一个 (x, y, z) 元组。
        :return: 一个字典，键是采样点的索引，值是该采样点的平均体素强度。
        """
    # 获取前景体素的坐标
    foreground_coords = np.array(np.where(mask))  # 形状为 (3, num_foreground)

    # 使用 KDTree 加速最近邻查询
    tree = KDTree(sampling_points)

    # 计算每个前景体素的最近采样点索引
    closest_sample_indices = tree.query(foreground_coords.T, k=1)[1].flatten()  # 获取最近的采样点索引

    # 存储每个采样点的关联体素
    associated_voxels = {i: [] for i in range(len(sampling_points))}

    # 将前景体素关联到采样点
    for i, sample_idx in enumerate(closest_sample_indices):
        associated_voxels[sample_idx].append(foreground_coords[:, i])

    # 计算每个采样点的平均体素强度
    average_intensities = []
    for sample_idx, voxel_coords in associated_voxels.items():
        intensities = [img[tuple(voxel)] for voxel in voxel_coords]  # 获取关联体素的强度
        # average_intensities[sample_idx] = np.mean(intensities) if intensities else 0.0
        average_intensities.append(np.mean(intensities) if intensities else 0.0)

    return average_intensities

def calc_surrounding_mean_intensity_file(img_file, mask_file, swc_file, temp_save_file):
    if(os.path.exists(temp_save_file)):
        result = np.load(temp_save_file, allow_pickle=True)
        return result['surrounding_mean_intensity_list'], result['path_dist_to_soma_list'], result['point_intensity_list']
    img = tifffile.imread(img_file)
    mask = tifffile.imread(mask_file)

    def generate_tree_from_swc_file(swc_file):
        swc = pd.read_csv(swc_file, sep=' ', header=None, comment='#')
        swc.columns = ['n', 'type', 'x', 'y', 'z', 'r', 'parent']
        G = nx.DiGraph()
        for i in range(swc.shape[0]):
            n = swc.iloc[i]['n']
            type = swc.iloc[i]['type']
            x = swc.iloc[i]['x']
            y = swc.iloc[i]['y']
            z = swc.iloc[i]['z']
            r = swc.iloc[i]['r']
            parent = swc.iloc[i]['parent']
            G.add_node(n, x=x, y=y, z=z, r=r, type=type, path_dist_to_soma=0)
            if parent != -1:
                G.add_edge(parent, n)
        return G

    swc_G = generate_tree_from_swc_file(swc_file)

    swc_point_list = []
    path_dist_to_soma_list = []
    point_intensity_list = []
    for node in swc_G.nodes:
        x, y, z = swc_G.nodes[node]['x'], swc_G.nodes[node]['y'], swc_G.nodes[node]['z']
        x, y, z = max(0, min(x, img.shape[2] - 1)), max(0, min(y, img.shape[1] - 1)), max(0, min(z, img.shape[0] - 1))
        swc_point_list.append((z, y, x))
        point_intensity_list.append(img[int(z), int(y), int(x)])
        parent = list(swc_G.predecessors(node))
        if(parent):
            parent = parent[0]
            dist_to_parent = np.sqrt((swc_G.nodes[node]['x'] - swc_G.nodes[parent]['x'])**2 +
                                     (swc_G.nodes[node]['y'] - swc_G.nodes[parent]['y'])**2 +
                                     (swc_G.nodes[node]['z'] - swc_G.nodes[parent]['z'])**2)
            swc_G.nodes[node]['path_dist_to_soma'] = swc_G.nodes[parent]['path_dist_to_soma'] + dist_to_parent
        path_dist_to_soma_list.append(swc_G.nodes[node]['path_dist_to_soma'])
    # print(np.sum(mask), np.sum(get_surrounding_mask(mask, iterations=2)))
    surrounding_mean_intensity_list = compute_mean_intensity_list(img,
                                                                  get_surrounding_mask(mask, iterations=1),
                                                                  swc_point_list)
    result = {
        'surrounding_mean_intensity_list': surrounding_mean_intensity_list,
        'path_dist_to_soma_list': path_dist_to_soma_list,
        "point_intensity_list": point_intensity_list
    }
    np.savez(temp_save_file, **result)
    return surrounding_mean_intensity_list, path_dist_to_soma_list, point_intensity_list

def plot_line(surrounding_mean_intensity_list, path_dist_to_soma_list):

    plt.plot(path_dist_to_soma_list, surrounding_mean_intensity_list, 'o')
    plt.show()
    plt.close()

def plot_trend_line(surrounding_intensity_trend_list, signal_intensity_trend_list):
    max_path_dist = max([x[0] for x in surrounding_intensity_trend_list])
    surrounding_intensity_trend = [ [] for _ in range(max_path_dist+1)]
    signal_intensity_trend = [ [] for _ in range(max_path_dist+1)]

    for path_dist, surrounding_intensity in surrounding_intensity_trend_list:
        surrounding_intensity_trend[path_dist].append(surrounding_intensity)
    for path_dist, signal_intensity in signal_intensity_trend_list:
        signal_intensity_trend[path_dist].append(signal_intensity)

    surrounding_intensity_trend = [np.mean(x) if x else 0 for x in surrounding_intensity_trend]
    signal_intensity_trend = [np.mean(x) if x else 0 for x in signal_intensity_trend]
    delta_intensity_trend = np.array(signal_intensity_trend) - np.array(surrounding_intensity_trend)

    plt.figure(figsize=(4, 4))
    x = np.linspace(0, 1, max_path_dist+1)
    plt.plot(x, signal_intensity_trend, label='signal intensity', color='red', alpha=0.5)
    plt.plot(x, surrounding_intensity_trend, label='surrounding intensity', color='blue', alpha=0.5)
    plt.plot(x, delta_intensity_trend, label='delta intensity', color='green', alpha=0.5)
    plt.legend()
    plt.show()
    plt.close()

def calc_surrounding_mean_intensity_dir(img_dir, mask_dir, swc_dir, temp_save_dir):
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]

    # Parallel(n_jobs=8)(delayed(calc_surrounding_mean_intensity_file)(
    #     os.path.join(img_dir, img_file),
    #     os.path.join(mask_dir, img_file),
    #     os.path.join(swc_dir, img_file.replace('.tif', '.swc'))) for img_file in tqdm(img_files)
    # )

    # Parallel(n_jobs=20)(delayed(calc_surrounding_mean_intensity_file)(
    #     os.path.join(img_dir, img_file),
    #     os.path.join(mask_dir, img_file),
    #     os.path.join(swc_dir, img_file.replace('.tif', '.swc')),
    #     os.path.join(temp_save_dir, img_file.replace('.tif', '.npz'))
    # ) for img_file in tqdm(img_files))

    surrounding_intensity_trend_list = []
    signal_intensity_trend_list = []

    for img_file in tqdm(img_files):
        surrounding_mean_intensity_list, path_dist_to_soma_list, point_intensity_list = calc_surrounding_mean_intensity_file(
            os.path.join(img_dir, img_file),
            os.path.join(mask_dir, img_file),
            os.path.join(swc_dir, img_file.replace('.tif', '.swc')),
            os.path.join(temp_save_dir, img_file.replace('.tif', '.npz'))
        )
        path_dist_to_soma_list = np.array(path_dist_to_soma_list).astype(np.float32)
        path_dist_to_soma_list = path_dist_to_soma_list / path_dist_to_soma_list.max() * 1000
        path_dist_to_soma_list = path_dist_to_soma_list.astype(np.int32)
        for surrounding_mean_intensity, path_dist_to_soma, point_intensity in zip(surrounding_mean_intensity_list, path_dist_to_soma_list, point_intensity_list):
            surrounding_intensity_trend_list.append((int(path_dist_to_soma), surrounding_mean_intensity))
            signal_intensity_trend_list.append((int(path_dist_to_soma), point_intensity))

    plot_trend_line(surrounding_intensity_trend_list, signal_intensity_trend_list,)


# rescale_mask_dir(mask_dir, rescaled_mask_dir)
# rescale_img_dir(source_img_dir, rescaled_img_dir)
calc_surrounding_mean_intensity_dir(rescaled_img_dir, rescaled_mask_dir, manual_swc_dir, temp_result_save_dir)


# estimate_radius_dir(manual_swc_dir, rescaled_mask_dir, manual_swc_with_radius_dir)







