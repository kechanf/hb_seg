import os
import pandas as pd
from networkx import neighbors
import numpy as np
from ecut.graph_cut import ECut
from ecut.swc_handler import parse_swc, write_swc
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import tempfile
from gcut.python.neuron_segmentation import NeuronSegmentation
import sys
import io
from contextlib import contextmanager
import time
import sys
sys.setrecursionlimit(1000000)

meta_file = r"/data2/kfchen/tracing_ws/human_neuron_recon_50k/meta_info_0205.xlsx"
meta_info = pd.read_excel(meta_file)

def find_nearest_swc_point(swc_point_list, target_pos, dist_limit=25): # 5um
    # min_dist, nearest_point_num = np.inf, None
    # for swc_point in range(len(swc_point_list)):
    #     swc_num, swc_x, swc_y, swc_z = swc_point_list.iloc[swc_point][['n', 'x', 'y', 'z']].values
    #     dist = (swc_x - target_pos[0]) ** 2 + (swc_y - target_pos[1]) ** 2 + (swc_z - target_pos[2]) ** 2
    #     if(min_dist > dist):
    #         min_dist = dist
    #         nearest_point_num = swc_num

    swc_points = swc_point_list[['x', 'y', 'z']].values
    distances = np.sum((swc_points - target_pos) ** 2, axis=1)
    min_dist_index = np.argmin(distances)
    min_dist = distances[min_dist_index]
    nearest_point_num = swc_point_list.iloc[min_dist_index]['n']

    if (min_dist > dist_limit ** 2):
        return None
    else:
        return nearest_point_num

def get_mapped_somas(swc_file=r"C:\Users\12626\Desktop\50k\swc\15533.swc", ptrs_dir=r"C:\Users\12626\Desktop\50k\swc\PTRSB_Somas", output_marker_file=None):
    if(os.path.exists(output_marker_file)):
        return pd.read_csv(output_marker_file, sep=',', header=None, names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g', 'color_b'])

    neuron_id = int(os.path.basename(swc_file).split('_')[0].split('.')[0])
    ptrs_files = os.listdir(ptrs_dir)
    doc_name, xy_resolution, z_resolution, soma_x, soma_y, soma_z = meta_info[meta_info['cell_id'] == neuron_id][
        ['document_name', 'xy_resolution', 'z_resolution', 'soma_x', 'soma_y', 'soma_z']].values[0]
    ptrs_file = [f for f in ptrs_files if doc_name in f][0]
    ptrs_file = os.path.join(ptrs_dir, ptrs_file)

    # print(ptrs_file)
    # x,y,z,radius,shape,name,comment,color_r,color_g,color_b
    ptrs_markers = pd.read_csv(ptrs_file, sep=',',
                               comment='#',
                               header=None,
                               names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
                                      'color_b'])
    # print(ptrs_markers['name'])

    current_soma_pos = ptrs_markers[ptrs_markers['name'] == neuron_id][['x', 'y', 'z']].values[0]
    # print(current_soma_pos)

    ptrs_markers['x'] = (ptrs_markers['x'] - current_soma_pos[0] + float(soma_x)) * float(xy_resolution) / 1000.0
    ptrs_markers['y'] = (ptrs_markers['y'] - current_soma_pos[1] + float(soma_y)) * float(xy_resolution) / 1000.0
    ptrs_markers['z'] = (ptrs_markers['z'] - current_soma_pos[2] + float(soma_z)) * float(z_resolution) / 1000.0

    # save
    ptrs_markers.to_csv(output_marker_file, index=False,  sep=',', header=False)

    return ptrs_markers


def get_soma_around_e_cut(swc_file=r"C:\Users\12626\Desktop\50k\swc\15533.swc", ptrs_dir=r"C:\Users\12626\Desktop\50k\swc\PTRSB_Somas", output_swc_file=None, output_marker_file=None):
    neuron_id = int(os.path.basename(swc_file).split('_')[1].split('.')[0])
    ptrs_files = os.listdir(ptrs_dir)
    doc_name, xy_resolution, z_resolution, soma_x, soma_y, soma_z = meta_info[meta_info['cell_id'] == neuron_id][
        ['document_name', 'xy_resolution', 'z_resolution', 'soma_x', 'soma_y', 'soma_z']].values[0]
    ptrs_file = [f for f in ptrs_files if doc_name in f][0]
    ptrs_file = os.path.join(ptrs_dir, ptrs_file)

    # print(ptrs_file)
    # x,y,z,radius,shape,name,comment,color_r,color_g,color_b
    ptrs_markers = pd.read_csv(ptrs_file, sep=',',
                               comment='#',
                               header=None,
                               names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
                                      'color_b'])
    # print(ptrs_markers['name'])

    current_soma_pos = ptrs_markers[ptrs_markers['name'] == neuron_id][['x', 'y', 'z']].values[0]
    # print(current_soma_pos)

    ptrs_markers['x'] = (ptrs_markers['x'] - current_soma_pos[0] + float(soma_x)) * float(xy_resolution) / 1000.0
    ptrs_markers['y'] = (ptrs_markers['y'] - current_soma_pos[1] + float(soma_y)) * float(xy_resolution) / 1000.0
    ptrs_markers['z'] = (ptrs_markers['z'] - current_soma_pos[2] + float(soma_z)) * float(z_resolution) / 1000.0

    swc_point_list = pd.read_csv(swc_file, sep=' ',
                                 comment='#',
                                 header=None, names=['n', 'type', 'x', 'y', 'z', 'r', 'pn'])
    # print(ptrs_markers[['x', 'y', 'z']])
    potential_somanum_in_swc = []
    for soma_in_markers in range(len(ptrs_markers)):
        # neighbor_soma_pos = ptrs_markers.iloc[soma_in_markers][['x', 'y', 'z']].values
        neighbor_somanum, neighbor_soma_x, neighbor_soma_y, neighbor_soma_z = ptrs_markers.iloc[soma_in_markers][['name', 'x', 'y', 'z']].values
        nearest_swc_num = find_nearest_swc_point(swc_point_list, (neighbor_soma_x, neighbor_soma_y, neighbor_soma_z))
        if(nearest_swc_num is not None):
            potential_somanum_in_swc.append([neighbor_somanum, int(nearest_swc_num)])

    print(potential_somanum_in_swc)

    if(neuron_id not in [f[0] for f in potential_somanum_in_swc]):
        print(f"Neuron {neuron_id} not found in the swc file.")
        return

    # NOTE: the node numbering of this tree should be SORTED, and starts from ZERO.
    tree = parse_swc(swc_file)
    tree_nodenum = [f[1] for f in potential_somanum_in_swc]
    begin_time = time.time()
    print("before cut:", len(tree))
    e = ECut(tree, tree_nodenum)  # 0 and 100 are the IDs of somata
    e.run()
    trees = e.export_swc()
    print("time:", time.time() - begin_time)
    for i in range(len(potential_somanum_in_swc)):
        if (potential_somanum_in_swc[i][0] == neuron_id):
            write_swc(trees[potential_somanum_in_swc[i][1]], output_swc_file)
            print("---Soma found.---")

def get_soma_around_g_cut(swc_file=r"C:\Users\12626\Desktop\50k\swc\15533.swc", ptrs_dir=r"C:\Users\12626\Desktop\50k\swc\PTRSB_Somas", output_swc_file=None, output_marker_file=None):
    neuron_id = int(os.path.basename(swc_file).split('_')[0].split('.')[0])
    ptrs_markers = get_mapped_somas(swc_file, ptrs_dir, output_marker_file)

    swc_point_list = pd.read_csv(swc_file, sep=' ',
                                 comment='#',
                                 header=None, names=['n', 'type', 'x', 'y', 'z', 'r', 'pn'])
    # print(ptrs_markers[['x', 'y', 'z']])
    potential_somanum_in_swc = []
    for soma_in_markers in range(len(ptrs_markers)):
        # neighbor_soma_pos = ptrs_markers.iloc[soma_in_markers][['x', 'y', 'z']].values
        neighbor_somanum, neighbor_soma_x, neighbor_soma_y, neighbor_soma_z = ptrs_markers.iloc[soma_in_markers][['name', 'x', 'y', 'z']].values
        nearest_swc_num = find_nearest_swc_point(swc_point_list, (neighbor_soma_x, neighbor_soma_y, neighbor_soma_z))
        if(nearest_swc_num is not None):
            potential_somanum_in_swc.append([neighbor_somanum, int(nearest_swc_num)])

    # print(potential_somanum_in_swc)
    current_neuron_soma = None
    if (neuron_id not in [f[0] for f in potential_somanum_in_swc]):
        print(f"Neuron {neuron_id} not found in the swc file.")
        print(f"{neuron_id}, Potential somas: {potential_somanum_in_swc}")
        return
    else:
        for i in range(len(potential_somanum_in_swc)):
            if (potential_somanum_in_swc[i][0] == neuron_id):
                current_neuron_soma = potential_somanum_in_swc[i][1]

    with tempfile.TemporaryDirectory() as temp_out_dir:
        # save markers
        temp_marker_file = os.path.join(temp_out_dir, os.path.basename(swc_file).replace('.swc', '_markers.txt'))
        with open(temp_marker_file, 'w') as f:
            for i in potential_somanum_in_swc:
                f.write(f"{i[1]}\n")

        @contextmanager
        def suppress_stdout():
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                yield
            finally:
                sys.stdout = original_stdout

        with suppress_stdout(): # 禁用内部输出
            segmentor = NeuronSegmentation(swc_file, temp_marker_file, scale_z=1,
                                           scale_ouput_z=False)
            segmentor.segment()
            segmentor.save(temp_out_dir)

        for f in os.listdir(temp_out_dir):
            # print(f)
            if f"_soma={current_neuron_soma}.swc" in f:
                # copy and rename
                shutil.copy(os.path.join(temp_out_dir, f), output_swc_file)

def my_neuron_pruning(swc_file, output_swc_file, output_marker_file=None):
    neuron_id = int(os.path.basename(swc_file).split('_')[0].split('.')[0])
    ptrs_markers = get_mapped_somas(swc_file, ptrs_dir, output_marker_file)
    swc_point_list = pd.read_csv(swc_file, sep=' ',
                                 comment='#',
                                 header=None, names=['n', 'type', 'x', 'y', 'z', 'r', 'pn'])
    # print(ptrs_markers[['x', 'y', 'z']])
    potential_somanum_in_swc = []
    for soma_in_markers in range(len(ptrs_markers)):
        # neighbor_soma_pos = ptrs_markers.iloc[soma_in_markers][['x', 'y', 'z']].values
        neighbor_somanum, neighbor_soma_x, neighbor_soma_y, neighbor_soma_z = ptrs_markers.iloc[soma_in_markers][
            ['name', 'x', 'y', 'z']].values
        nearest_swc_num = find_nearest_swc_point(swc_point_list, (neighbor_soma_x, neighbor_soma_y, neighbor_soma_z))
        if (nearest_swc_num is not None and (not neighbor_somanum == neuron_id)):
            potential_somanum_in_swc.append(int(nearest_swc_num))

    print(potential_somanum_in_swc)

    def traverse_tree(node, children, prune_list, remaining_points, new_id_map):
        """递归遍历树，跳过 prune_list 中的点及其子树"""
        if node in prune_list:
            # print(node, prune_list, (node in prune_list))
            return  # 跳过当前点及其子树
        # 重新编号
        new_id = len(new_id_map) + 1
        new_id_map[node] = new_id
        remaining_points.append(node)
        if node in children:
            for child in children[node]:
                traverse_tree(child, children, prune_list, remaining_points, new_id_map)


    def save_swc(file_path, points):
        """将点保存为 SWC 文件"""
        with open(file_path, 'w') as f:
            for p in points:
                f.write(f"{p['n']} {p['type']} {p['x']} {p['y']} {p['z']} {p['r']} {p['pn']}\n")

    def parse_swc(file_path):
        """解析 SWC 文件，返回点列表和父子关系"""
        points = []
        point_dict = {}  # 存储 n 到点的映射
        children = {}  # 存储每个点的子节点
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue  # 跳过注释行
                n, type_, x, y, z, r, pn = map(float, line.strip().split())
                point = {'n': int(n), 'type': int(type_), 'x': x, 'y': y, 'z': z, 'r': r, 'pn': int(pn)}
                points.append(point)
                point_dict[int(n)] = point  # 将 n 映射到点
                if pn not in children:
                    children[pn] = []
                children[pn].append(int(n))
        return points, point_dict, children

    def renumber_points(points, point_dict, remaining_points, new_id_map, children):
        """重新编号剩余的点，并生成新的 SWC 数据"""
        new_points = []
        for node in remaining_points:
            point = point_dict[node]  # 直接通过字典查找点
            pn = point['pn']
            new_pn = new_id_map[pn] if pn != -1 else -1  # soma 点的 pn 保持为 -1
            new_points.append({
                'n': new_id_map[node],
                'type': point['type'],
                'x': point['x'],
                'y': point['y'],
                'z': point['z'],
                'r': point['r'],
                'pn': new_pn
            })
        return new_points

    def prune_swc(input_file, output_file, prune_list):
        """主函数：修剪 SWC 文件并保存结果"""
        # 解析 SWC 文件
        points, point_dict, children = parse_swc(input_file)

        # 找到 soma 点（pn = -1 的点）
        soma_node = next(p['n'] for p in points if p['pn'] == -1)

        # 遍历树，跳过 prune_list 中的点及其子树
        remaining_points = []  # 存储剩余的点
        new_id_map = {}  # 存储旧 ID 到新 ID 的映射
        traverse_tree(soma_node, children, prune_list, remaining_points, new_id_map)

        # 重新编号剩余的点
        new_points = renumber_points(points, point_dict, remaining_points, new_id_map, children)

        # 保存结果
        save_swc(output_file, new_points)

    prune_swc(swc_file, output_swc_file, potential_somanum_in_swc)



def try_get_soma_around(swc_file=r"C:\Users\12626\Desktop\50k\swc\15533.swc", ptrs_dir=r"C:\Users\12626\Desktop\50k\swc\PTRSB_Somas", output_swc_file=None, output_marker_file=None):
    # get_soma_around_g_cut(swc_file, ptrs_dir, output_swc_file, output_marker_file)
    # return
    try:
        get_soma_around_g_cut(swc_file, ptrs_dir, output_swc_file, output_marker_file)
    except Exception as e:
        print(f"Error in e-cut: {swc_file}, {e}")

# source_swc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/8_estimated_radius_swc"
source_swc_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_swcs"
# source_swc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/7_scaled_1um_swc"
# target_swc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/9_e_cut_swc"
target_swc_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_swcs_g_cut"
output_markers_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/mapped_markers"
os.makedirs(target_swc_dir, exist_ok=True)
os.makedirs(output_markers_dir, exist_ok=True)

ptrs_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas"
swc_files = os.listdir(source_swc_dir)

# for f in tqdm(swc_files, desc="ecut/gcut swcs"):
#     try:
#         my_neuron_pruning(
#             os.path.join(source_swc_dir, f),
#             os.path.join(target_swc_dir, f),
#             os.path.join(output_markers_dir, f.replace('.swc', '.txt'))
#         )
#     except Exception as e:
#         print(f"Error in e-cut: {f}, {e}")

for f in tqdm(swc_files, desc="ecut/gcut swcs"):
    try_get_soma_around(
        os.path.join(source_swc_dir, f),
        ptrs_dir,
        os.path.join(target_swc_dir, f),
        os.path.join(output_markers_dir, f.replace('.swc', '.txt'))
    )

# Parallel(n_jobs=8)(delayed(try_get_soma_around)(
#     os.path.join(source_swc_dir, f),
#     ptrs_dir,
#     os.path.join(target_swc_dir, f),
#     os.path.join(output_markers_dir, f.replace('.swc', '.txt'))
# ) for f in tqdm(swc_files, desc="ecut/gcut swcs"))
