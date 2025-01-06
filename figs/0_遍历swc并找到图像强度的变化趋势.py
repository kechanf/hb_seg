import os.path

import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import joblib
import seaborn as sns
import sys
import networkx as nx
import tempfile
import shutil
import subprocess
import cv2
from scipy.fftpack import fftn, fftshift
from brokenaxes import brokenaxes

v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"
mouse_neuron_info_file = "/data/kfchen/trace_ws/img_noise_test/seu1876/41467_2024_54745_MOESM3_ESM.xlsx"
mouse_neuron_info_df = pd.read_excel(mouse_neuron_info_file)
neuron_info_df = pd.read_csv("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv", encoding='gbk')
N_JOBS = 20
sys.setrecursionlimit(50000)

# 读取SWC文件到pandas DataFrame
def read_swc(file_path, xy_resolution):
    # 读取SWC文件
    df = pd.read_csv(file_path, sep=' ', header=None, comment='#',
                     names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'],
                     dtype={'id': int, 'type': int, 'x': float, 'y': float, 'z': float, 'radius': float, 'parent': int})
    df['x'] = df['x'] * xy_resolution / 1000
    df['y'] = df['y'] * xy_resolution / 1000
    return df

def read_eswc(file_path, xy_resolution):
    ##n,type,x,y,z,radius,parent,seg_id,level,mode,timestamp,teraflyindex,feature_value
    df = pd.read_csv(file_path, sep=' ', header=None, comment='#',
                     names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent', 'seg_id', 'level', 'mode', 'timestamp', 'teraflyindex', 'feature_value'],
                     dtype={'id': int, 'type': int, 'x': float, 'y': float, 'z': float, 'radius': float, 'parent': int,
                            'seg_id': int, 'level': int, 'mode': int, 'timestamp': int, 'teraflyindex': int, 'feature_value': float})
    df['x'] = df['x'] * xy_resolution / 1000
    df['y'] = df['y'] * xy_resolution / 1000
    return df

# 从soma节点开始遍历所有节点
def traverse_from_soma(swc_df, img):
    # 找到soma节点（通常是type == 1）
    soma_node = swc_df[swc_df['type'] == 1].iloc[0]
    soma_id = soma_node['id']

    # 创建一个字典来存储从每个节点到其子节点的连接关系
    tree = {}
    for _, row in swc_df.iterrows():
        if row['parent'] != -1:  # parent == -1表示没有父节点（根节点）
            if row['parent'] not in tree:
                tree[row['parent']] = []
            tree[row['parent']].append(row['id'])

    # 存储每个节点到soma的路径距离和直线距离
    distance_to_soma = {soma_id: 0.0}  # soma到自己的距离为0
    straight_line_distance = {soma_id: 0.0}  # soma到自己的直线距离为0
    img_value = {soma_id: img[int(soma_node['z']), int(soma_node['y']), int(soma_node['x'])]}
    visited = set()  # 记录已访问的节点

    # 计算两个节点之间的欧氏距离
    def euclidean_distance(p1, p2):
        return np.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2 + (p1['z'] - p2['z']) ** 2)

    # 深度优先搜索DFS，从soma开始遍历
    def dfs(node_id, current_distance, current_straight_distance):
        # 遍历该节点的所有子节点
        if node_id in visited:
            return

        visited.add(node_id)

        # 获取当前节点的信息
        current_node = swc_df[swc_df['id'] == node_id].iloc[0]

        # 记录当前节点到soma的路径距离（路径总和）和直线距离
        distance_to_soma[node_id] = current_distance
        straight_line_distance[node_id] = current_straight_distance
        z, y, x = int(current_node['z']), int(current_node['y']), int(current_node['x'])
        z, y, x = min(max(z, 0), img.shape[0]-1), min(max(y, 0), img.shape[1]-1), min(max(x, 0), img.shape[2]-1)
        img_value[node_id] = img[z, y, x]

        # 遍历所有子节点
        if node_id in tree:
            for child_id in tree[node_id]:
                # 计算从当前节点到子节点的直线距离
                child_node = swc_df[swc_df['id'] == child_id].iloc[0]
                edge_distance = euclidean_distance(current_node, child_node)
                # 递归调用DFS，累加路径距离和直线距离
                dfs(child_id, current_distance + edge_distance, euclidean_distance(soma_node, child_node))

    # 从soma节点开始遍历
    dfs(soma_id, 0.0, 0.0)

    swc_df['path_dist'] = np.nan
    swc_df['euclidean_dist'] = np.nan
    swc_df['image_intensity'] = np.nan
    for node_id in distance_to_soma:
        swc_df.loc[swc_df['id'] == node_id, 'path_dist'] = distance_to_soma[node_id]
        swc_df.loc[swc_df['id'] == node_id, 'euclidean_dist'] = straight_line_distance[node_id]
        swc_df.loc[swc_df['id'] == node_id, 'image_intensity'] = img_value[node_id]

    # return distance_to_soma, straight_line_distance, img_value
    return swc_df

def traverse_from_soma_eswc(swc_df):
    # 找到soma节点（通常是type == 1）
    soma_node = swc_df[swc_df['type'] == 1].iloc[0]
    soma_id = soma_node['id']

    # 创建一个字典来存储从每个节点到其子节点的连接关系
    tree = {}
    for _, row in swc_df.iterrows():
        if row['parent'] != -1:  # parent == -1表示没有父节点（根节点）
            if row['parent'] not in tree:
                tree[row['parent']] = []
            tree[row['parent']].append(row['id'])

    # 存储每个节点到soma的路径距离和直线距离
    distance_to_soma = {soma_id: 0.0}  # soma到自己的距离为0
    straight_line_distance = {soma_id: 0.0}  # soma到自己的直线距离为0
    img_value = {soma_id: soma_node['level']}
    visited = set()  # 记录已访问的节点

    # 计算两个节点之间的欧氏距离
    def euclidean_distance(p1, p2):
        return np.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2 + (p1['z'] - p2['z']) ** 2)

    # 深度优先搜索DFS，从soma开始遍历
    # def dfs(node_id, current_distance, current_straight_distance):
    #     # 遍历该节点的所有子节点
    #     if node_id in visited:
    #         return
    #
    #     visited.add(node_id)
    #
    #     # 获取当前节点的信息
    #     current_node = swc_df[swc_df['id'] == node_id].iloc[0]
    #
    #     # 记录当前节点到soma的路径距离（路径总和）和直线距离
    #     distance_to_soma[node_id] = current_distance
    #     straight_line_distance[node_id] = current_straight_distance
    #     img_value[node_id] = current_node['level']
    #
    #     # 遍历所有子节点
    #     if node_id in tree:
    #         for child_id in tree[node_id]:
    #             # 计算从当前节点到子节点的直线距离
    #             child_node = swc_df[swc_df['id'] == child_id].iloc[0]
    #             edge_distance = euclidean_distance(current_node, child_node)
    #             # 递归调用DFS，累加路径距离和直线距离
    #             dfs(child_id, current_distance + edge_distance, euclidean_distance(soma_node, child_node))
    #
    # # 从soma节点开始遍历
    # dfs(soma_id, 0.0, 0.0)
    def dfs_optimized(soma_id):
        # 通过将swc_df转化为字典，提高查找效率
        node_info = {row['id']: row for _, row in swc_df.iterrows()}

        # 初始化栈以模拟递归
        stack = [(soma_id, 0.0, 0.0)]  # (当前节点id, 当前路径距离, 当前直线距离)

        # 用一个集合记录访问的节点，避免重复访问
        visited = set()

        while stack:
            node_id, current_distance, current_straight_distance = stack.pop()

            # 如果节点已经访问过，跳过
            if node_id in visited:
                continue

            visited.add(node_id)

            # 获取当前节点的信息
            current_node = node_info[node_id]

            # 记录当前节点到soma的路径距离（路径总和）和直线距离
            distance_to_soma[node_id] = current_distance
            straight_line_distance[node_id] = current_straight_distance
            img_value[node_id] = current_node['level']

            # 遍历所有子节点
            if node_id in tree:
                for child_id in tree[node_id]:
                    # 获取子节点信息
                    child_node = node_info[child_id]
                    # 计算当前节点到子节点的直线距离
                    edge_distance = euclidean_distance(current_node, child_node)
                    # 将子节点和新计算的距离压入栈中
                    stack.append(
                        (child_id, current_distance + edge_distance, euclidean_distance(soma_node, child_node)))

    # 从soma节点开始遍历
    dfs_optimized(soma_id)

    # swc_df['path_dist'] = np.nan
    # swc_df['euclidean_dist'] = np.nan
    # swc_df['image_intensity'] = np.nan
    swc_df.loc[:, 'path_dist'] = np.nan
    swc_df.loc[:, 'euclidean_dist'] = np.nan
    swc_df.loc[:, 'image_intensity'] = np.nan
    for node_id in distance_to_soma:
        swc_df.loc[swc_df['id'] == node_id, 'path_dist'] = distance_to_soma[node_id]
        swc_df.loc[swc_df['id'] == node_id, 'euclidean_dist'] = straight_line_distance[node_id]
        swc_df.loc[swc_df['id'] == node_id, 'image_intensity'] = img_value[node_id]
    # swc_df['path_dist'] = swc_df['id'].map(distance_to_soma)
    # swc_df['euclidean_dist'] = swc_df['id'].map(straight_line_distance)
    # swc_df['image_intensity'] = swc_df['id'].map(img_value)

    # return distance_to_soma, straight_line_distance, img_value
    return swc_df

def plot_histogram(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['path_dist'], df['image_intensity'], alpha=0.5, c=df['image_intensity'], cmap='viridis')
    plt.xlabel('Path Distance to Soma')
    plt.ylabel('Image Intensity')
    plt.title('Path Distance to Soma vs Image Intensity')
    # plt.colorbar(label='Image Intensity')
    plt.show()
    plt.close()

def calc_dist_i_file(swc_file, img_file, neuron_info_df, save_dir="/data/kfchen/trace_ws/img_noise_test/extended_swc"):
    save_file = os.path.join(save_dir, os.path.basename(swc_file).replace('.swc', '.csv'))
    if os.path.exists(save_file):
        swc_df = pd.read_csv(save_file)
        return swc_df['path_dist'], swc_df['image_intensity']

    id = int(os.path.basename(swc_file).split('.')[0])
    xy_resolution = neuron_info_df.loc[neuron_info_df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0]
    img = tifffile.imread(img_file).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = resize(img, (img.shape[0], img.shape[1] * xy_resolution / 1000, img.shape[2] * xy_resolution / 1000), order=2)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    # img = np.flip(img, axis=1)

    # 读取SWC文件
    swc_df = read_swc(swc_file, xy_resolution)
    swc_df = traverse_from_soma(swc_df, img)
    swc_df.to_csv(save_file, index=False)

    return swc_df['path_dist'], swc_df['image_intensity']
    # plot_histogram(swc_df)

def calc_dist_i_file_v2(swc_file, img_file, save_file):
    if os.path.exists(save_file):
        swc_df = pd.read_csv(save_file)
        return swc_df['path_dist'], swc_df['image_intensity']

    img = tifffile.imread(img_file).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = np.flip(img, axis=1)

    swc_df = pd.read_csv(swc_file, sep=' ', header=None, comment='#',
                     names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'],
                     dtype={'id': int, 'type': int, 'x': float, 'y': float, 'z': float, 'radius': float, 'parent': int})

    # mip = np.max(img, axis=0)
    # plt.imshow(mip)
    # plt.savefig(os.path.join("/data/kfchen/trace_ws/img_noise_test/temp_mip", os.path.basename(swc_file).replace('.swc', '_img.png')))
    # plt.close()
    # # cv2
    # for x, y, z in zip(swc_df['x'], swc_df['y'], swc_df['z']):
    #     cv2.circle(mip, (int(x), int(y)), 1, 255, -1)
    # plt.imshow(mip)
    # plt.savefig(os.path.join("/data/kfchen/trace_ws/img_noise_test/temp_mip", os.path.basename(swc_file).replace('.swc', '_swc.png')))
    # plt.close()
    # exit()

    try:
        swc_df = traverse_from_soma(swc_df, img)
    except:
        print(f"Error in {swc_file}")
        swc_df['path_dist'] = np.nan
        swc_df['image_intensity'] = np.nan

    swc_df.to_csv(save_file, index=False)

    return swc_df['path_dist'], swc_df['image_intensity']

def crop_box_from_soma(swc_file, lim):
    x_lim, y_lim, z_lim = lim
    df = pd.read_csv(swc_file, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
    soma = df[df.type == 1 & (df.pid == -1)]
    assert len(soma) == 1
    soma = soma.iloc[0]
    soma_x, soma_y, soma_z = soma[['x', 'y', 'z']]

    x_min, x_max = soma_x - x_lim/2, soma_x + x_lim/2
    y_min, y_max = soma_y - y_lim/2, soma_y + y_lim/2
    z_min, z_max = soma_z - z_lim/2, soma_z + z_lim/2
    df_crop = df[(df.x >= x_min) & (df.x <= x_max) &
                 (df.y >= y_min) & (df.y <= y_max) &
                 (df.z >= z_min) & (df.z <= z_max)]
    df_crop.x = df_crop.x - x_min
    df_crop.y = df_crop.y - y_min
    df_crop.z = df_crop.z - z_min
    return df_crop

def estimate_radius(img_file, swc_file, out_file):
    def v3d_get_radius(img_path, swc_path, out_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            # 获取文件名
            img_filename = os.path.basename(img_path).split('_')[0] + '.tif'
            swc_filename = os.path.basename(swc_path).split('_')[0] + '.swc'
            output_filename = os.path.basename(out_path).split('_')[0] + '.swc'

            # 设置缓存文件路径
            img_cache_path = os.path.join(temp_dir, img_filename)
            swc_cache_path = os.path.join(temp_dir, swc_filename)
            out_cache_path = os.path.join(temp_dir, output_filename)

            # 将文件复制到缓存路径
            shutil.copy(img_path, img_cache_path)
            shutil.copy(swc_path, swc_cache_path)

            # 设置命令字符串
            radius2d = 1
            cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x neuron_radius -f neuron_radius -i {img_cache_path} {swc_cache_path} -o {out_cache_path} -p 40 {radius2d}'
            cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')

            # 执行命令
            # print(f"Running command: {cmd_str}")
            subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)

            # 将结果从临时路径复制到实际输出路径
            shutil.copy(out_cache_path, out_path)

    def load_swc_to_undirected_graph(swc_file_path):
        """从SWC文件加载数据，构建无向图，并记录每个节点的parent信息"""
        df = pd.read_csv(swc_file_path, delim_whitespace=True, comment='#', header=None,
                         names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'])
        G = nx.Graph()

        for _, row in df.iterrows():
            # 添加节点，同时记录parent信息
            G.add_node(row['id'], pos=(row['x'], row['y'], row['z']), radius=row['radius'], type=row['type'],
                       parent=row['parent'])
            if row['parent'] != -1:
                G.add_edge(row['parent'], row['id'])

        return G

    def find_nearest_node(G, target_pos):
        """ 在图中找到与给定坐标最近的节点 """
        nearest_node = None
        min_distance = float('inf')

        for node in G.nodes(data=True):
            pos = node[1]['pos']
            distance = np.linalg.norm(np.array(pos) - np.array(target_pos))
            if distance < min_distance:
                nearest_node = node[0]
                min_distance = distance

        return nearest_node

    def export_to_swc_dfs(G, root_pos, output_filename):
        if(os.path.exists(output_filename)):
            os.remove(output_filename)

        start_node = find_nearest_node(G, root_pos)

        # 调整根节点
        potential_root = max(G.nodes, key=lambda x: G.degree(x))
        potential_root_degree = G.degree(potential_root)
        potential_root_list = [node for node in G.nodes if G.degree(node) == potential_root_degree]
        for node in potential_root_list:
            if G.degree(node) > 4 and len(potential_root_list) == 1:  # 这个点的度数大于4
                start_node = node
            elif (nx.shortest_path_length(G, start_node, node) < 3):
                start_node = node
            elif (np.linalg.norm(np.array(G.nodes[node]['pos']) - np.array(root_pos)) < 10):
                start_node = node

        # 打开文件进行写入
        with open(output_filename, 'w') as f:
            # 写入SWC文件的头部注释
            f.write("# SWC file generated from DFS traversal\n")
            f.write("# Columns: id type x y z radius parent\n")

            # 用于存储节点的新编号和访问状态
            new_id = 1
            visited = set()
            stack = [(start_node, -1)]  # (current_node, parent_id_in_new_swc)

            while stack:
                node, parent_id = stack.pop()
                if node not in visited:
                    visited.add(node)
                    node_data = G.nodes[node]
                    pos = node_data['pos']
                    radius = node_data['radius']
                    if (parent_id == -1):
                        node_type = 1
                    else:
                        node_type = 3

                    # 写入当前节点数据
                    f.write(f"{new_id} {node_type} {pos[0]} {pos[1]} {pos[2]} {radius} {parent_id}\n")

                    # 更新父节点ID为当前节点的新ID
                    current_parent_id = new_id
                    new_id += 1

                    # 将所有未访问的邻接节点添加到栈中
                    for neighbor in G.neighbors(node):
                        if neighbor not in visited:
                            stack.append((neighbor, current_parent_id))

    def calc_node_dist(G, node1, node2):
        pos1 = np.array(G.nodes[node1]['pos'])
        pos2 = np.array(G.nodes[node2]['pos'])
        return np.linalg.norm(pos1 - pos2)

    def gaussian_smoothing_radius_tree(G, sigma=1.0):
        smoothed_values = {}
        soma_r = G.nodes[1]['radius']
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            weights = []
            values = []
            for neighbor in neighbors:
                distance = calc_node_dist(G, node, neighbor)
                weight = np.exp(- (distance ** 2) / (2 * sigma ** 2))
                weights.append(weight)
                values.append(G.nodes[neighbor]['radius'])
            # 自身的权重
            self_weight = np.exp(0)
            total_weight = self_weight + sum(weights)
            weighted_sum = G.nodes[node]['radius'] * self_weight + sum(w * v for w, v in zip(weights, values))
            smoothed_values[node] = weighted_sum / total_weight
        nx.set_node_attributes(G, smoothed_values, 'radius')
        G.nodes[1]['radius'] = soma_r
        return G

    def smoothing_swc_file(swc_file_path, output_filename):
        G = load_swc_to_undirected_graph(swc_file_path)
        G = gaussian_smoothing_radius_tree(G)
        root_pos = G.nodes[1]['pos']
        # print(root_pos)
        export_to_swc_dfs(G, root_pos, output_filename)

    if(os.path.exists(out_file)):
        return
    try:
        v3d_get_radius(img_file, swc_file, out_file)
        smoothing_swc_file(out_file, out_file)
    except:
        print(f"Error in {swc_file}")

def swc_to_img(img_file, swc_file, mask_file):
    img = tifffile.imread(img_file)
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i {swc_file} ' \
              f'-p {img.shape[2]} {img.shape[1]} {img.shape[0]} -o {mask_file}'
    cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
    # print(cmd_str)
    subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
    mask = tifffile.imread(mask_file)
    mask = mask.astype(np.float32)
    mask = np.flip(mask, axis=1)
    mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
    tifffile.imwrite(mask_file, mask.astype('uint8'))

def compute_forground_info(img_file, mask_file, save_file):
    if(os.path.exists(save_file)):
        return np.load(save_file)
    # 读取图像和掩膜
    img = tifffile.imread(img_file)
    mask = tifffile.imread(mask_file)
    mask = np.where(mask > 0, 255, 0)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) # * 255

    # 确保掩膜是二值的（前景为1，背景为0）
    foreground_mask = (mask == 255)
    background_mask = (mask == 0)

    # 计算前景和背景的强度中位数
    foreground_intensities = img[foreground_mask]
    background_intensities = img[background_mask]

    foreground_median = np.median(foreground_intensities) if foreground_intensities.size > 0 else None
    background_median = np.median(background_intensities) if background_intensities.size > 0 else None
    foreground_mean = np.mean(foreground_intensities) if foreground_intensities.size > 0 else None
    background_mean = np.mean(background_intensities) if background_intensities.size > 0 else None
    contrast = (foreground_mean - background_mean) / np.sqrt(foreground_intensities.var() + background_intensities.var())
    contrast_guo = foreground_median - background_median
    foreground_homogeneity_entropy = -np.sum(foreground_intensities * np.log(foreground_intensities + 1e-6)) / (foreground_intensities.size + 1e-6)
    background_homogeneity_entropy = -np.sum(background_intensities * np.log(background_intensities + 1e-6)) / (background_intensities.size + 1e-6)
    foreground_uniformity = -np.sum(foreground_intensities * foreground_intensities) / (foreground_intensities.size + 1e-6)
    background_uniformity = -np.sum(background_intensities * background_intensities) / (background_intensities.size + 1e-6)

    result = {
        'foreground_median': foreground_median,
        'background_median': background_median,
        'foreground_mean': foreground_mean,
        'background_mean': background_mean,
        'contrast': contrast,
        'contrast_guo': contrast_guo,
        'foreground_homogeneity_entropy': foreground_homogeneity_entropy,
        'background_homogeneity_entropy': background_homogeneity_entropy,
        'foreground_uniformity': foreground_uniformity,
        'background_uniformity': background_uniformity
    }

    np.save(save_file, result)
    return result

def prepare_proposed():
    img_dir = "/data/kfchen/trace_ws/to_gu/img"
    swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/2_flip_after_sort"
    swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]

    # resize
    target_img_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/cropped_img_1um"
    target_swc_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/cropped_swc_1um"

    def current_resize(img_file, swc_file, target_img_file, target_swc_file):
        id = int(os.path.basename(swc_file).split('.')[0])
        xy_resolution = neuron_info_df.loc[neuron_info_df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0]
        xy_resolution = xy_resolution / 1000

        if(not os.path.exists(target_img_file)):
            img = tifffile.imread(img_file).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = resize(img, (img.shape[0], img.shape[1] * xy_resolution, img.shape[2] * xy_resolution), order=2)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            tifffile.imwrite(target_img_file, img.astype('uint8'))

        if(not os.path.exists(target_swc_file)):
            swc = pd.read_csv(swc_file, sep=' ', header=None, comment='#',
                              names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'],
                              dtype={'id': int, 'type': int, 'x': float, 'y': float, 'z': float, 'radius': float,
                                     'parent': int})
            swc.x = swc.x * xy_resolution
            swc.y = swc.y * xy_resolution
            swc.to_csv(target_swc_file, sep=' ', header=False, index=False)

    swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
    # if(os.path.exists(target_img_dir) == False):
    #     os.makedirs(target_img_dir, exist_ok=True)
    #     os.makedirs(target_swc_dir, exist_ok=True)
    #     joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(current_resize)(os.path.join(img_dir, swc_file.replace('.swc', '.tif')),
    #                                                               os.path.join(swc_dir, swc_file),
    #                                                               os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
    #                                                               os.path.join(target_swc_dir, swc_file)) for swc_file in tqdm(swc_files))

    swc_with_radius_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/cropped_swc_1um_with_radius"
    if(os.path.exists(swc_with_radius_dir) == False):
        os.makedirs(swc_with_radius_dir, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(estimate_radius)(os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
                                                                   os.path.join(target_swc_dir, swc_file),
                                                                   os.path.join(swc_with_radius_dir, swc_file)) for swc_file in tqdm(swc_files))

    mask_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/cropped_mask_1um"
    if(os.path.exists(mask_dir) == False):
        os.makedirs(mask_dir, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(swc_to_img)(os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
                                                              os.path.join(swc_with_radius_dir, swc_file),
                                                              os.path.join(mask_dir, swc_file.replace('.swc', '.tif'))) for swc_file in tqdm(swc_files))
    # return
    ex_swc_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/extended_swc"
    if(os.path.exists(ex_swc_dir) == False):
        os.makedirs(ex_swc_dir, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(calc_dist_i_file_v2)(os.path.join(swc_with_radius_dir, swc_file),
                                                                      os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
                                                                      os.path.join(ex_swc_dir, swc_file.replace('.swc', '.csv'))) for swc_file in tqdm(swc_files))

    foreground_info_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/foreground_info"
    if(os.path.exists(foreground_info_dir) == False):
        os.makedirs(foreground_info_dir, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(compute_forground_info)(os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
                                                                              os.path.join(mask_dir, swc_file.replace('.swc', '.tif')),
                                                                              os.path.join(foreground_info_dir, swc_file.replace('.swc', '.npy'))) for swc_file in tqdm(swc_files))



def prepare_seu1876(crop_lim=(512, 512, 512)):
    # source_swc_dir = "/data/kfchen/trace_ws/quality_control_test/mouse/seu1876/raw"
    # target_swc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/" + f"cropped_box_swc_{crop_lim[0]}_{crop_lim[1]}_{crop_lim[2]}"
    #
    # if(os.path.exists(target_swc_dir) == False):
    #     os.makedirs(target_swc_dir, exist_ok=True)
    #     swc_files = [f for f in os.listdir(source_swc_dir) if f.endswith('.swc')]
    #
    #     def current_task(swc_file, source_swc_dir, target_swc_dir, crop_lim):
    #         df = crop_box_from_soma(os.path.join(source_swc_dir, swc_file), crop_lim)
    #         df.to_csv(os.path.join(target_swc_dir, swc_file), sep=' ', header=False)
    #     # for swc_file in swc_files:
    #     #     df = crop_box_from_soma(os.path.join(source_swc_dir, swc_file), crop_lim)
    #     #     df.to_csv(os.path.join(target_swc_dir, swc_file), sep=' ', header=False)
    #     joblib.Parallel(n_jobs=10)(joblib.delayed(current_task)(swc_file, source_swc_dir, target_swc_dir, crop_lim) for swc_file in tqdm(swc_files))

    profile_swc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/profiled_final"
    # target_swc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/" + f"cropped_box_swc_{crop_lim[0]}_{crop_lim[1]}_{crop_lim[2]}"
    extended_swc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/extended_swc_from_eswc"
    os.makedirs(extended_swc_dir, exist_ok=True)
    mouse_neuron_info_file = "/data/kfchen/trace_ws/img_noise_test/seu1876/41467_2024_54745_MOESM3_ESM.xlsx"
    mouse_neuron_info_df = pd.read_excel(mouse_neuron_info_file)
    fail_list = []

    eswc_files = [f for f in os.listdir(profile_swc_dir) if f.endswith('.eswc')]

    def current_task(eswc_file, profile_swc_dir, extended_swc_dir, crop_lim, mouse_neuron_info_df):
        image_id = eswc_file.split('_')[0]
        if(image_id == 'pre'):
            image_id = eswc_file.split('_')[1]
        xy_resolution = mouse_neuron_info_df.loc[mouse_neuron_info_df['Image ID'] == int(image_id), 'Resolution_XY (𝜇𝑚/voxel)'].values[0]
        target_file = os.path.join(extended_swc_dir, eswc_file.replace('.eswc', '.csv'))
        if os.path.exists(target_file):
            return
        df = read_eswc(os.path.join(profile_swc_dir, eswc_file), float(xy_resolution)*1000)
        soma = df[df.type == 1 & (df.parent == -1)]
        assert len(soma) == 1
        soma = soma.iloc[0]
        soma_x, soma_y, soma_z = soma[['x', 'y', 'z']]
        x_min, x_max = soma_x - crop_lim[0]/2, soma_x + crop_lim[0]/2
        y_min, y_max = soma_y - crop_lim[1]/2, soma_y + crop_lim[1]/2
        z_min, z_max = soma_z - crop_lim[2]/2, soma_z + crop_lim[2]/2
        df_crop = df[(df.x >= x_min) & (df.x <= x_max) &
                     (df.y >= y_min) & (df.y <= y_max) &
                     (df.z >= z_min) & (df.z <= z_max)]
        df_crop = traverse_from_soma_eswc(df_crop)
        df_crop.to_csv(target_file, index=False)

    # print(len(eswc_files))
    for eswc_file in eswc_files:
        image_id = eswc_file.split('_')[0]
        if(image_id == 'pre'):
            image_id = eswc_file.split('_')[1]
        # print(image_id)
        try:
            xy_resolution = mouse_neuron_info_df.loc[mouse_neuron_info_df['Image ID'] == int(image_id), 'Resolution_XY (𝜇𝑚/voxel)'].values[0]
        except:
            fail_list.append(eswc_file)

    # print(len(fail_list), len(eswc_files))
    eswc_files = [f for f in eswc_files if f not in fail_list]

    joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(current_task)(eswc_file, profile_swc_dir, extended_swc_dir, crop_lim, mouse_neuron_info_df) for eswc_file in tqdm(eswc_files))
    # for eswc_file in eswc_files:
    #     current_task(eswc_file, profile_swc_dir, extended_swc_dir, crop_lim, mouse_neuron_info_df)


def prepare_seu1876_new():
    # estimate_radius
    target_img_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/cropped_img_1um"
    target_swc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/cropped_swc_1um"

    swc_files = [f for f in os.listdir(target_swc_dir) if f.endswith('.swc')]
    swc_with_radius_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/cropped_swc_1um_with_radius"
    if(os.path.exists(swc_with_radius_dir) == False):
        os.makedirs(swc_with_radius_dir, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(estimate_radius)(os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
                                                                   os.path.join(target_swc_dir, swc_file),
                                                                   os.path.join(swc_with_radius_dir, swc_file)) for swc_file in tqdm(swc_files))
    mask_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/cropped_mask_1um"
    if(os.path.exists(mask_dir) == False):
        os.makedirs(mask_dir, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(swc_to_img)(os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
                                                              os.path.join(swc_with_radius_dir, swc_file),
                                                              os.path.join(mask_dir, swc_file.replace('.swc', '.tif'))) for swc_file in tqdm(swc_files))
    # return
    # calc_dist_i_file_v2
    ex_human_swc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/extended_swc"
    # if(os.path.exists(ex_human_swc_dir) == False):
    os.makedirs(ex_human_swc_dir, exist_ok=True)
#     # for swc_file in tqdm(swc_files):
#     #     calc_dist_i_file_v2(os.path.join(target_swc_dir, swc_file),
#     #                         os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
#     #                         os.path.join(ex_human_swc_dir, swc_file.replace('.swc', '.csv')))
    joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(calc_dist_i_file_v2)(os.path.join(target_swc_dir, swc_file),
                                                                os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
                                                                os.path.join(ex_human_swc_dir, swc_file.replace('.swc', '.csv'))) for swc_file in tqdm(swc_files))

    foreground_info_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/foreground_info"
    if(os.path.exists(foreground_info_dir) == False):
        os.makedirs(foreground_info_dir, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(compute_forground_info)(os.path.join(target_img_dir, swc_file.replace('.swc', '.tif')),
                                                                              os.path.join(mask_dir, swc_file.replace('.swc', '.tif')),
                                                                              os.path.join(foreground_info_dir, swc_file.replace('.swc', '.npy'))) for swc_file in tqdm(swc_files))
def plt_fig1(num_bins=1000):
    ex_human_swc_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/extended_swc"
    # ex_mouse_eswc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/extended_swc_from_eswc"
    ex_mouse_eswc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/extended_swc"

    human_files = [f for f in os.listdir(ex_human_swc_dir) if f.endswith('.csv')]
    mouse_files = [f for f in os.listdir(ex_mouse_eswc_dir) if f.endswith('.csv')]

    path_dists = [[], []]
    image_intensities = [[], []]
    for human_file in human_files:
        df = pd.read_csv(os.path.join(ex_human_swc_dir, human_file))
        current_path_dist, current_image_intensity = df['path_dist'], df['image_intensity']
        # current_path_dist = [int(i) for i in current_path_dist]
        current_path_dist = current_path_dist / np.max(current_path_dist) * num_bins
        current_path_dist = [int(i) for i in current_path_dist]
        current_image_intensity = (current_image_intensity - current_image_intensity.min()) / (
                    current_image_intensity.max() - current_image_intensity.min()) * 255
        path_dists[0].extend(current_path_dist)
        image_intensities[0].extend(current_image_intensity)

    for mouse_file in mouse_files:
        df = pd.read_csv(os.path.join(ex_mouse_eswc_dir, mouse_file))[['type', 'path_dist', 'image_intensity']]
        df = df[df['type'] != 2]
        df = df.dropna()

        current_path_dist, current_image_intensity = df['path_dist'], df['image_intensity']
        current_path_dist = current_path_dist / np.max(current_path_dist) * num_bins
        current_path_dist = [int(i) for i in current_path_dist]
        current_image_intensity = (current_image_intensity - current_image_intensity.min()) / (
                    current_image_intensity.max() - current_image_intensity.min()) * 255
        if (len(current_image_intensity) == 0 or current_image_intensity[0] < 255 * 0.5):
            continue
        path_dists[1].extend(current_path_dist)
        image_intensities[1].extend(current_image_intensity)

    # hist
    # 设置清晰度
    plt.rcParams['figure.dpi'] = 300
    set2_colors = plt.cm.get_cmap('Set2').colors
    plt.figure(figsize=(4, 3))
    df = pd.DataFrame({
        'path_dist': np.array(path_dists[0]).astype(np.float32) / num_bins, #float(path_dists[0])/num_bins,
        'image_intensity': image_intensities[0]
    })
    human_average_intensities = df.groupby('path_dist')['image_intensity'].mean().reset_index()
    df = pd.DataFrame({
        'path_dist': np.array(path_dists[1]).astype(np.float32) / num_bins,
        'image_intensity': image_intensities[1]
    })
    mouse_average_intensities = df.groupby('path_dist')['image_intensity'].mean().reset_index()
    # plt.scatter(human_average_intensities['path_dist'], human_average_intensities['image_intensity'], alpha=0.5, color='red')
    # plt.scatter(mouse_average_intensities['path_dist'], mouse_average_intensities['image_intensity'], alpha=0.5, color='blue')
    # 折线图
    # plt.plot(human_average_intensities['path_dist'], human_average_intensities['image_intensity'], color='darkorange')
    # plt.plot(mouse_average_intensities['path_dist'], mouse_average_intensities['image_intensity'], color='skyblue')
    # 拟合曲线

    # def exp_decay(x, A, B, C):
    #     return A * np.exp(-B * x) + C
    #
    #
    # # 拟合人类数据
    # human_x = human_average_intensities['path_dist']
    # human_y = human_average_intensities['image_intensity']
    # params_human, _ = curve_fit(exp_decay, human_x, human_y, p0=[1, 0.1, 0])  # 初始猜测值
    #
    # # 拟合小鼠数据
    # mouse_x = mouse_average_intensities['path_dist']
    # mouse_y = mouse_average_intensities['image_intensity']
    # params_mouse, _ = curve_fit(exp_decay, mouse_x, mouse_y, p0=[1, 0.1, 0])  # 初始猜测值
    #
    # human_fit_y = exp_decay(human_x, *params_human)
    # plt.plot(human_x, human_fit_y, label="Human Fit", color='darkorange', linewidth=2)
    #
    # # 绘制小鼠拟合曲线
    # mouse_fit_y = exp_decay(mouse_x, *params_mouse)
    # plt.plot(mouse_x, mouse_fit_y, label="Mouse Fit", color='skyblue', linewidth=2)

    # 折线

    plt.plot(human_average_intensities['path_dist'], human_average_intensities['image_intensity'], color=set2_colors[0])
    plt.plot(mouse_average_intensities['path_dist'], mouse_average_intensities['image_intensity'], color=set2_colors[1])
    # print(human_average_intensities['image_intensity'][-10:])
    # print(mouse_average_intensities['image_intensity'][-10:])

    plt.xlim(0, 1)
    plt.ylim(0, 255)

    # plt.scatter(average_intensities['path_dist'], average_intensities['image_intensity'], alpha=0.5, c=average_intensities['image_intensity'], cmap='viridis')
    # kde
    # sns.kdeplot(x=path_dists, y=image_intensities, cmap='viridis', shade=True, cbar=True)

    plt.xlabel('Norm. path dist. to soma', fontsize=15)
    plt.ylabel('Voxel value', fontsize=15)
    # tick
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.title('Path Distance to Soma vs Image Intensity')
    # legend
    plt.legend(['Dye-injection', 'Genetic labeling'], fontsize=12, frameon=False)
    plt.tight_layout()
    # plt.colorbar(label='Image Intensity')
    # plt.show()
    plt.savefig("/data/kfchen/trace_ws/img_noise_test/Path_Distance_to_Soma_vs_Image_Intensity.png")
    plt.close()

    # plt.scatter(path_dists, image_intensities, alpha=0.5, c=image_intensities, cmap='viridis')
    # 拟合曲线
    # z = np.polyfit(path_dists, image_intensities, 1)
    # p = np.poly1d(z)
    # plt.plot(path_dists, p(path_dists), "r--")
    # plt.xlabel('Path Distance to Soma')
    # plt.ylabel('Image Intensity')
    # plt.title('Path Distance to Soma vs Image Intensity')
    # # plt.colorbar(label='Image Intensity')
    # plt.savefig("/data/kfchen/trace_ws/img_noise_test/Path_Distance_to_Soma_vs_Image_Intensity.png")
    # plt.close()

def plt_fig2():
    plt.rcParams['figure.dpi'] = 300
    human_foreground_info_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/foreground_info"
    mouse_foreground_info_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/foreground_info"

    human_files = [f for f in os.listdir(human_foreground_info_dir) if f.endswith('.npy')]
    mouse_files = [f for f in os.listdir(mouse_foreground_info_dir) if f.endswith('.npy')]

    # human_sbc, mouse_sbc = [], []
    human_result_list = {
        'foreground_median': [],
        'background_median': [],
        'foreground_mean': [],
        'background_mean': [],
        'contrast': [],
        'contrast_guo': [],
        'foreground_homogeneity_entropy': [],
        'background_homogeneity_entropy': [],
        'foreground_uniformity': [],
        'background_uniformity': [],
    }
    mouse_result_list = {
        'foreground_median': [],
        'background_median': [],
        'foreground_mean': [],
        'background_mean': [],
        'contrast': [],
        'contrast_guo': [],
        'foreground_homogeneity_entropy': [],
        'background_homogeneity_entropy': [],
        'foreground_uniformity': [],
        'background_uniformity': [],
    }


    for human_file in human_files:
        current_result = np.load(os.path.join(human_foreground_info_dir, human_file), allow_pickle=True).item()
        for key in human_result_list:
            human_result_list[key].append(current_result[key])


    for mouse_file in mouse_files:
        current_result = np.load(os.path.join(mouse_foreground_info_dir, mouse_file), allow_pickle=True).item()
        for key in mouse_result_list:
            mouse_result_list[key].append(current_result[key])


    # plot violin
    fig, ax = plt.subplots(5, 2, figsize=(8, 10))
    ax = ax.flatten()
    for i, key in enumerate(human_result_list):
        sns.violinplot(data=[human_result_list[key], mouse_result_list[key]], ax=ax[i])
        ax[i].set_xticks([0, 1])
        ax[i].set_xticklabels(['Human', 'Mouse'])
        ax[i].set_title(key)


    plt.tight_layout()
    plt.savefig("/data/kfchen/trace_ws/img_noise_test/SBC.png")
    plt.close()

def crop_swc_files():
    img_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/cropped_img_1um"
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    def current_task(img_file):
        img = tifffile.imread(os.path.join(img_dir, img_file))
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = np.flip(img, axis=1)
        tifffile.imwrite(os.path.join(img_dir, img_file), img.astype('uint8'))
    # for img_file in img_files:
    #     current_task(img_file)
    joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(current_task)(img_file) for img_file in tqdm(img_files))

def plt_fig3():
    mask_source = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/mask"
    mask_target = "/data/kfchen/trace_ws/img_noise_test/proposed/my_great_mask"

    mask_files = [f for f in os.listdir(mask_source) if f.endswith('.tif')]
    def current_task(mask_file, mask_source, mask_target):
        id = mask_file.split('.')[0]
        xy_resolution = neuron_info_df.loc[neuron_info_df.iloc[:, 0] == int(id), 'xy拍摄分辨率(*10e-3μm/px)'].values[0]
        xy_resolution = float(xy_resolution) / 1000
        mask = tifffile.imread(os.path.join(mask_source, mask_file))
        mask = mask.astype(np.float32)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = resize(mask, (mask.shape[0], mask.shape[1] * xy_resolution, mask.shape[2] * xy_resolution), order=1)
        mask = np.where(mask > 0.5, 255, 0)
        mask = np.flip(mask, axis=1)
        tifffile.imwrite(os.path.join(mask_target, mask_file), mask.astype('uint8'))
    # for mask_file in mask_files:
    #     current_task(mask_file)
    if(os.path.exists(mask_target) == False):
        os.makedirs(mask_target, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(current_task)(mask_file, mask_source, mask_target) for mask_file in tqdm(mask_files))

    # 计算前景和背景的直方图
    mask_files = [f for f in os.listdir(mask_target) if f.endswith('.tif')]
    img_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/cropped_img_1um"
    temp_save_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/hist_temp"
    # os.makedirs(temp_save_dir, exist_ok=True)

    total_foreground_hist = [0 for _ in range(256)]
    total_background_hist = [0 for _ in range(256)]

    def current_task(mask_file, img_dir, mask_target, temp_save_dir):
        save_file = os.path.join(temp_save_dir, mask_file.replace('.tif', '.npy'))
        if(os.path.exists(save_file)):
            return np.load(save_file)
        img = tifffile.imread(os.path.join(img_dir, mask_file))
        mask = tifffile.imread(os.path.join(mask_target, mask_file))
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

        foreground_mask = (mask == 255)
        background_mask = (mask == 0)

        foreground_intensities = img[foreground_mask]
        background_intensities = img[background_mask]

        foreground_hist, _ = np.histogram(foreground_intensities, bins=256, range=(0, 255), density=False)
        background_hist, _ = np.histogram(background_intensities, bins=256, range=(0, 255), density=False)

        # save
        np.save(save_file, (foreground_hist, background_hist))

        return foreground_hist, background_hist

    if(os.path.exists(temp_save_dir) == False):
        os.makedirs(temp_save_dir, exist_ok=True)
        joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(current_task)(mask_file, img_dir, mask_target, temp_save_dir) for mask_file in tqdm(mask_files))

    for mask_file in mask_files:
        foreground_hist, background_hist = np.load(os.path.join(temp_save_dir, mask_file.replace('.tif', '.npy')))
        total_size = np.sum(foreground_hist) + np.sum(background_hist)
        foreground_hist = foreground_hist
        background_hist = background_hist
        total_foreground_hist = [a + b for a, b in zip(total_foreground_hist, foreground_hist)]
        total_background_hist = [a + b for a, b in zip(total_background_hist, background_hist)]

    print(np.sum(total_foreground_hist))
    print(np.sum(total_background_hist))
    total_foreground_hist = total_foreground_hist / np.sum(total_foreground_hist)
    total_background_hist = total_background_hist / np.sum(total_background_hist)

    set2_colors = plt.cm.get_cmap('Set2').colors

    plt.rcParams['figure.dpi'] = 300
    # plot
    plt.figure(figsize=(4, 3))
    plt.plot(total_foreground_hist, color=set2_colors[2])
    plt.plot(total_background_hist, color=set2_colors[3])
    plt.yscale('log')
    # plt.ylim(-0.0005, 0.008)
    # plt.ylim(0.0192, 0.020)
    plt.xlabel('Voxel value', fontsize=15)
    plt.ylabel('Proportion', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    # 关闭上边框和右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.title('Intensity Histogram')
    plt.legend(['Foreground', 'Background'], frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig("/data/kfchen/trace_ws/img_noise_test/Intensity_Histogram.png")
    plt.close()


def plot_fig4(bins=100):

    img_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/cropped_img_1um"
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    # img_file = img_files[0]

    def current_task(img_file, bins, save_file):
        if(os.path.exists(save_file)):
            return
            return np.load(save_file)
        try:
            img = tifffile.imread(os.path.join(img_dir, img_file))
            fft_result = fftn(img)
            fft_shifted = fftshift(fft_result)
            energy_distribution = np.abs(fft_shifted) ** 2

            # 4. 将能量分布展平并计算直方图
            energy_flattened = energy_distribution.flatten()
            # hist, _ = np.histogram(energy_flattened, bins=bins, range=(energy_flattened.min(), energy_flattened.max()))
            # print(hist)
            np.save(save_file, energy_flattened)
            return
            return energy_flattened
        except Exception as e:
            print(e, img_file)
            return

    temp_save_dir = "/data/kfchen/trace_ws/img_noise_test/proposed/fft_temp"
    # if(os.path.exists(temp_save_dir) == False):
    os.makedirs(temp_save_dir, exist_ok=True)
    joblib.Parallel(n_jobs=N_JOBS)(joblib.delayed(current_task)(img_file, bins, os.path.join(temp_save_dir, img_file.replace('.tif', '.npy')) ) for img_file in tqdm(img_files))

    global_min, global_max = float('inf'), float('-inf')
    print("First Pass")
    for img_file in tqdm(img_files):
        try:
            energy = np.load(os.path.join(temp_save_dir, img_file.replace('.tif', '.npy')), allow_pickle=True)
            global_min = min(global_min, energy.min())
            global_max = max(global_max, energy.max())
        except:
            pass

    print(f"Global Min: {global_min}, Global Max: {global_max}")

    # 初始化直方图
    total_hist = np.zeros(bins, dtype=np.int64)
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    print("Second Pass")
    # 第二遍：逐文件计算直方图并累加
    for img_file in tqdm(img_files):
        try:
            energy = np.load(os.path.join(temp_save_dir, img_file.replace('.tif', '.npy')), allow_pickle=True)
            hist, _ = np.histogram(energy, bins=bin_edges)
            total_hist += hist
        except:
            pass

    set2_colors = plt.cm.get_cmap('Set2').colors
    # 清晰度
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(4, 4))
    plt.hist(total_hist, bins=bins, color='blue', alpha=0.7, log=True)
    plt.title("Histogram of Energy Distribution in Frequency Domain", fontsize=14)
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Frequency (Log Scale)", fontsize=12)
    # plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("/data/kfchen/trace_ws/img_noise_test/Energy_Histogram.png")
    plt.close()

    # # 5. 绘制直方图
    # plt.figure(figsize=(10, 6))
    # plt.hist(energy_flattened, bins=bins, color='blue', alpha=0.7, log=True)
    # plt.title("Histogram of Energy Distribution in Frequency Domain", fontsize=14)
    # plt.xlabel("Energy", fontsize=12)
    # plt.ylabel("Frequency (Log Scale)", fontsize=12)
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.show()
    # plt.close()


# 主程序
if __name__ == '__main__':
    # crop_swc_files()
    # exit()

    # prepare_seu1876()
    # prepare_proposed()
    # prepare_seu1876_new()

    plt_fig1() # ok
    # plt_fig2()
    plt_fig3() # ok
    # plot_fig4()

