from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import tifffile
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import ball
from tqdm import tqdm
from human_neuron_recon_50k.trace.merge_seg_soma import expand_soma_to_origin_size
from scipy.spatial import KDTree
from math import acos, degrees
from skimage.transform import resize
from joblib import Parallel, delayed

meta_info_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/meta_0324.xlsx"
meta_info = pd.read_excel(meta_info_file)

def load_swc_to_undirected_graph(swc_file_path, resolution=(1, 1, 1)):
    """从SWC文件加载数据，构建无向图，并记录每个节点的parent信息"""
    df = pd.read_csv(swc_file_path, delim_whitespace=True, comment='#', header=None,
                     names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'])
    G = nx.Graph()

    for _, row in df.iterrows():
        # 添加节点，同时记录parent信息
        x, y, z = float(row['x']) * resolution[0], float(row['y']) * resolution[1], float(row['z']) * resolution[2]
        r = float(row['radius']) * (resolution[0] + resolution[1] + resolution[2]) / 3
        G.add_node(row['id'], pos=(x, y, z), radius=r, type=row['type'])
        if row['parent'] != -1:
            G.add_edge(row['parent'], row['id'])

    return G



def visualize_graph(G):
    # 可视化图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for node, data in G.nodes(data=True):
        x, y, z = data['pos']
        ax.scatter(x, y, z, s=100 * data['radius'])
    for u, v in G.edges():
        x = [G.nodes[n]['pos'][0] for n in (u, v)]
        y = [G.nodes[n]['pos'][1] for n in (u, v)]
        z = [G.nodes[n]['pos'][2] for n in (u, v)]
        ax.plot(x, y, z, 'k-')
    # plt.show()


def is_tree(G):
    # 检查图是否是连通的
    if not nx.is_connected(G.to_undirected()):
        # print("The graph is not connected.")
        return False

    # 检查图是否包含环
    if nx.is_tree(G):
        # print("The graph is a tree.")
        return True
    else:
        # print("The graph is not a tree; it has cycles.")
        return False

def go_find_nearest_node(G, target_pos):
    nearest_node = None
    min_distance = float('inf')

    for node in G.nodes(data=True):
        pos = node[1]['pos']
        distance = np.linalg.norm(np.array(pos) - np.array(target_pos))
        if distance < min_distance:
            nearest_node = node[0]
            min_distance = distance

    return nearest_node

def export_to_swc_dfs(G, start_node, output_filename, resolution=(1,1,1)):
    # start_node = find_nearest_node(G, root_pos)
    #
    # # 调整根节点
    # potential_root = max(G.nodes, key=lambda x: G.degree(x))
    # potential_root_degree = G.degree(potential_root)
    # potential_root_list = [node for node in G.nodes if G.degree(node) == potential_root_degree]
    # for node in potential_root_list:
    #     if G.degree(node) > 4 and len(potential_root_list) == 1: # 这个点的度数大于4
    #         start_node = node
    #     elif(nx.shortest_path_length(G, start_node, node) < 3):
    #         start_node = node
    #     elif(np.linalg.norm(np.array(G.nodes[node]['pos']) - np.array(root_pos)) < 10):
    #         start_node = node

    # 打开文件进行写入
    with open(output_filename, 'w') as f:
        # 写入SWC文件的头部注释
        f.write("# SWC file generated from DFS traversal\n")
        f.write("# Columns: id type x y z radius parent\n")

        # 用于存储节点的新编号和访问状态
        new_id = 1
        visited = set()
        stack = [(start_node, -1)]  # (current_node, parent_id_in_new_swc)
        soma_r = None

        while stack:
            node, parent_id = stack.pop()
            if node not in visited:
                visited.add(node)
                node_data = G.nodes[node]
                pos = node_data['pos']
                radius = node_data['radius']
                if(parent_id == -1):
                    node_type = 1
                else:
                    node_type = 3

                # 写入当前节点数据
                x, y, z = pos
                x, y, z = x/resolution[0], y/resolution[1], z/resolution[2]
                r = radius / ((resolution[0] + resolution[1] + resolution[2]) / 3)
                if(soma_r is None):soma_r = r
                else:
                    r = min(r, soma_r)
                f.write(f"{new_id} {node_type} {x} {y} {z} {r} {parent_id}\n")

                # 更新父节点ID为当前节点的新ID
                current_parent_id = new_id
                new_id += 1

                # 将所有未访问的邻接节点添加到栈中
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        stack.append((neighbor, current_parent_id))

def disconnect_with_shadow_nodes(graph):
    """
    1. 断开所有度数≥3的节点
    2. 对每个分支点生成n个影子节点（n为原度数）
    3. 将每个原连接分配给独立的影子节点
    （处理后所有连通分量都将成为简单路径，节点度数仅为1或2）
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("输入必须是networkx的Graph对象")

    G = deepcopy(graph)
    original_nodes = set(G.nodes())

    branching_points = [node for node, degree in dict(G.degree()).items()
                        if degree >= 3]

    shadow_nodes = {}
    connection_plan = {}

    # node_num = len(G.nodes()) !!!!!
    node_num = max(G.nodes())

    for node in branching_points:
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)

        # 生成n个影子节点（每个连接分配一个独立影子节点）
        origin_pos, origin_radius, origin_type = G.nodes[node]['pos'], G.nodes[node]['radius'], G.nodes[node]['type']
        shadow_nodes[node] = []  # 影子节点列表，包含原节点
        for i in range(0, degree):
            node_num = node_num + 1
            G.add_node(node_num, pos=origin_pos, radius=origin_radius, type=origin_type)
            shadow_nodes[node].append(node_num)

        # 断开原节点所有连接 删除原节点
        G.remove_node(node)

        # 将每个原连接分配给独立的影子节点
        for shadow, neighbor in zip(shadow_nodes[node], neighbors):
            G.add_edge(shadow, neighbor)
            # 输出degree
            # print(f"Node {shadow} connected to {neighbor}")
            # print(f"shadow degree:{G.degree(shadow)}, neighbor degree:{G.degree(neighbor)}")

    return G

# class GraphProcesser:


class GenerateSTEMChain:
    """
    处理仅包含简单链的图，根据mask规则修改图结构

    功能：
    1. 删除完全在mask中的链
    2. 保留完全不在mask中的链
    3. 对部分在mask中的链：
       - 删除mask中的节点
       - 将剩余子链连接到最近的质心点

    示例：
    >>> processor = GenerateSTEMChain()
    >>> G = nx.Graph([(1,2), (2,3), (4,5)])
    >>> mask = {1:True, 2:False, 3:True, 4:False, 5:False}
    >>> processor.process(G, mask, centroid_n=99)
    """

    def __init__(self):
        self._mask: Optional[Dict[int, bool]] = None
        self._centroid_n: Optional[int] = None

    def process(self,
                G: nx.Graph,
                mask: Dict[int, bool]) -> nx.Graph:
        """
        主处理方法

        参数：
        G -- 仅包含简单链的图
        mask -- {节点ID: 是否在mask中}
        centroid_n -- 质心节点ID（必须已存在于图中）

        返回：
        修改后的图（原地修改）
        """

        # centroid_n = len(G.nodes()) + 1
        # add a new node
        centroid_n = max(G.nodes()) + 1
        pos, radius = self._compute_centroid_and_max_sphere(mask)
        G.add_node(centroid_n, pos=(pos[2], pos[1], pos[0]), radius=radius, type=1)

        self._validate_input(G, mask, centroid_n)
        self._mask = mask
        self._centroid_n = centroid_n

        components = list(nx.connected_components(G))
        for chain_nodes in components:
            self._process_chain(G, chain_nodes)

        return G, centroid_n

    def _compute_centroid_and_max_sphere(self, mask_3d):
        """
        计算三维mask的质心及可放置的最大内接球半径

        参数：
        mask_3d : numpy.ndarray
            三维二值mask数组，True/1表示前景，False/0表示背景

        返回：
        centroid : tuple
            质心坐标 (z, y, x)
        max_radius : float
            质心位置可放置的最大球体半径（以体素为单位）
        """

        def compute_tolerant_max_sphere(mask_3d, centroid=None, tolerance=0.9, max_iter=100):
            """
            计算容忍部分背景体素的最大内接球半径

            参数：
            mask_3d : np.ndarray
                三维二值mask，True/1表示前景
            centroid : tuple, optional
                指定质心坐标 (z,y,x)，默认自动计算
            tolerance : float
                可接受的前景体素占比阈值（0-1）
            max_iter : int
                最大迭代次数（精度控制）

            返回：
            optimal_radius : float
                满足容忍条件的最大半径
            coverage : float
                实际达到的前景覆盖率
            """
            # 输入验证
            if not isinstance(mask_3d, np.ndarray) or mask_3d.ndim != 3:
                raise ValueError("输入必须是三维numpy数组")
            if not 0 < tolerance <= 1:
                raise ValueError("容忍度必须在(0,1]范围内")

            # 计算或验证质心
            if centroid is None:
                labeled = ndimage.label(mask_3d)[0]
                props = regionprops(labeled)
                if not props:
                    raise ValueError("mask中未找到前景区域")
                centroid = tuple(round(c) for c in props[0].centroid)
            else:
                if not all(0 <= c < s for c, s in zip(centroid, mask_3d.shape)):
                    raise ValueError("质心坐标超出图像范围")

            # 获取距离变换图（从质心出发）
            distance_map = ndimage.distance_transform_edt(mask_3d)
            max_possible = distance_map[centroid].item()

            # 二分法搜索满足条件的最佳半径
            low, high = 0, max_possible
            optimal_radius = 0
            best_coverage = 0

            # for _ in tqdm(range(max_iter), desc="搜索最佳半径"):
            for _ in range(max_iter):
                mid = (low + high) / 2
                coverage = evaluate_sphere_coverage(mask_3d, centroid, mid)

                if coverage >= tolerance:
                    optimal_radius = mid
                    best_coverage = coverage
                    low = mid  # 尝试更大的半径
                else:
                    high = mid  # 需要缩小半径

            return optimal_radius, best_coverage

        def evaluate_sphere_coverage(mask, center, radius):
            """
            评估球体内的前景体素占比

            参数：
            mask : np.ndarray
                输入三维mask
            center : tuple
                球心坐标 (z,y,x)
            radius : float
                球体半径

            返回：
            coverage : float
                球体内前景体素占比
            """
            # 创建球形结构元素
            r = int(np.ceil(radius))
            structure = ball(r).astype(bool)
            # print(structure.shape, r)

            # 计算需要裁剪的边界
            cz, cy, cx = center

            # 获取局部区域
            z_start, z_end = max(0, cz - r), min(mask.shape[0], cz + r)
            y_start, y_end = max(0, cy - r), min(mask.shape[1], cy + r)
            x_start, x_end = max(0, cx - r), min(mask.shape[2], cx + r)
            local_region = mask[z_start:z_end, y_start:y_end, x_start:x_end]

            # print(structure.shape, local_region.shape)
            # 调整结构元素大小以完全匹配局部区域
            struct_cropped = structure[:local_region.shape[0],
                             :local_region.shape[1],
                             :local_region.shape[2]]

            # 确保形状一致
            assert local_region.shape == struct_cropped.shape, \
                f"Shape mismatch: local_region {local_region.shape}, struct_cropped {struct_cropped.shape}"

            # 计算覆盖比例
            masked = local_region & struct_cropped
            total_voxels = struct_cropped.sum()
            if total_voxels == 0:
                return 0.0
            return masked.sum() / total_voxels

        def compute_global_centroid(mask_3d):
            """
            计算三维mask中所有前景点的全局质心

            参数：
            mask_3d : numpy.ndarray
                三维二值mask，True/1表示前景，False/0表示背景

            返回：
            centroid : tuple
                全局质心坐标 (z, y, x)
            """
            # 方法1：直接通过mask坐标计算（推荐）
            foreground_coords = np.argwhere(mask_3d)
            if len(foreground_coords) == 0:
                raise ValueError("mask中未找到前景区域")
            centroid = tuple(round(c) for c in foreground_coords.mean(axis=0))  # (z, y, x)

            return centroid

        # 输入验证
        if not isinstance(mask_3d, np.ndarray) or mask_3d.ndim != 3:
            raise ValueError("输入必须是三维numpy数组")

        centroid = compute_global_centroid(mask_3d)
        # print(f"质心坐标: {centroid}")
        radius, _ = compute_tolerant_max_sphere(mask_3d, centroid=centroid)
        # print(f"最大半径: {radius}")

        return centroid, radius

    def _validate_input(self,
                        G: nx.Graph,
                        mask: Dict[int, bool],
                        centroid_n: int):
        """输入验证"""
        if not isinstance(G, nx.Graph):
            raise TypeError("输入必须是networkx.Graph对象")
        if centroid_n not in G:
            raise ValueError(f"质心节点 {centroid_n} 不存在于图中")
        if any(d > 2 for _, d in G.degree()):
            raise ValueError("输入图必须仅包含简单链（所有节点度数≤2）")

    def _check_node_in_mask(self, G, node):
        x, y, z = G.nodes[node]['pos']
        x, y, z = int(x), int(y), int(z)
        x = min(max(x, 0), self._mask.shape[2] - 1)
        y = min(max(y, 0), self._mask.shape[1] - 1)
        z = min(max(z, 0), self._mask.shape[0] - 1)

        return self._mask[z, y, x] > 0

    def _process_chain(self, G: nx.Graph, chain_nodes: Set[int]):
        # print(f"Processing chain with {len(chain_nodes)} nodes")
        """处理单个链"""
        ordered_chain = self._get_ordered_chain(G, chain_nodes)
        if not ordered_chain:
            return

        # mask_status = [self._mask[node] for node in ordered_chain]
        mask_status = [self._check_node_in_mask(G, node) for node in ordered_chain]

        # 检查
        # print(self._centroid_n in ordered_chain, self._centroid_n in G)
        if(self._centroid_n in ordered_chain):
            # print(ordered_chain)
            return

        if all(mask_status):  # 情况1：全在mask中
            if (len(ordered_chain) == 1 and ordered_chain[0] == self._centroid_n):
                return
            ep = [ordered_chain[0], ordered_chain[-1]]
            if(ep[0] == ep[1]):
                ep = [ep[0]]
            for node in ordered_chain:
                if(node in ep):
                    G.add_edge(node, self._centroid_n)
                else:
                    G.remove_node(node)
            # G.remove_nodes_from(ordered_chain - set(ep))
        elif not any(mask_status):  # 情况2：全不在mask中
            return
        else:  # 情况3：部分在mask中
            self._process_partial_chain(G, ordered_chain)

    def _get_ordered_chain(self,
                           G: nx.Graph,
                           nodes: Set[int]) -> Optional[List[int]]:
        """将连通分量转换为有序链"""
        if len(nodes) == 1:
            return [next(iter(nodes))]

        endpoints = [n for n in nodes if G.degree(n) == 1]
        if len(endpoints) != 2:
            raise ValueError("链的端点数量不正确，必须为2个")

        start = endpoints[0]
        visited = set()
        ordered = []
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            ordered.append(node)
            for neighbor in G.neighbors(node):
                if neighbor in nodes and neighbor not in visited:
                    queue.append(neighbor)

        return ordered if len(ordered) == len(nodes) else None

    def _process_partial_chain(self,
                               G: nx.Graph,
                               chain: List[int]):
        """处理部分在mask中的链"""
        # 分割子链
        sub_chains = self._split_chain(G, chain)

        # 删除mask中的节点
        # nodes_to_remove = [n for n in chain if self._mask[n]]
        nodes_to_remove = [n for n in chain if self._check_node_in_mask(G, n)]
        G.remove_nodes_from(nodes_to_remove)

        # 连接子链到质心
        for sub_chain in sub_chains:
            if not sub_chain:
                continue
            self._connect_subchain(G, sub_chain)

    def _split_chain(self, G,  chain: List[int]) -> List[List[int]]:
        """根据mask分割链为连续的非mask子链"""
        sub_chains = []
        current = []

        for node in chain:
            if not self._check_node_in_mask(G, node):
                current.append(node)
            else:
                if current:
                    sub_chains.append(current)
                    current = []

        if current:
            sub_chains.append(current)

        return sub_chains

    def _connect_subchain(self,
                          G: nx.Graph,
                          sub_chain: List[int]):
        """将子链连接到质心"""
        # endpoints = [sub_chain[0], sub_chain[-1]]
        distances = {}
        endpoints = [sub_chain[0], sub_chain[-1]]
        for ep in endpoints:
            # distances[ep] = nx.shortest_path_length(G, ep, self._centroid_n)
            # 直线距离
            # print(ep, self._centroid_n, len(G.nodes))
            distances[ep] = np.linalg.norm(np.array(G.nodes[ep]['pos']) - np.array(G.nodes[self._centroid_n]['pos']))

        closest = min(endpoints, key=lambda x: distances[x])
        G.add_edge(closest, self._centroid_n)

class ChainAdder:
    def __init__(self, graph, root, mask, distance_threshold=1, angle_threshold=60, radius_ratio_threshold=1.5):
        """
        初始化链处理器

        参数：
        graph : networkx.Graph
            包含初始的树和游离的简单链
        root : int
        distance_threshold : float
            连接距离阈值
        angle_threshold : float
            连接角度阈值（单位：度）
        radius_ratio_threshold : float
            连接半径比率阈值(父子节点)，即子节点半径不能大于父节点的radius_ratio_threshold倍
        """

        self.dist_thresh = distance_threshold
        self.angle_thresh = angle_threshold
        self.radius_ratio_thresh = radius_ratio_threshold
        # self._build_kdtree()
        self.parent = {}  # 记录每个节点的父节点
        self.in_tree_nodes = []
        self.root = root
        self._mask = mask

        self._get_tree_and_parent(graph, root)

        components = list(nx.connected_components(graph))
        chains = [chain_nodes for chain_nodes in components if(root not in chain_nodes)]
        # get ordered
        chains = [self._get_ordered_chain(graph, chain_nodes) for chain_nodes in chains]

        self.run(graph, chains, self.dist_thresh, self.angle_thresh, self.radius_ratio_thresh)

        self._merge_nearby_nodes(graph)
        self._remove_chain_in_mask(graph)



    # def _build_kdtree(self):
    #     """构建图的节点空间索引"""
    #     self.graph_coords = np.array([self.graph.nodes[node]['pos'] for node in self.graph.nodes()])
    #     self.kdtree = KDTree(self.graph_coords)

    def _check_node_in_mask(self, G, node):
        x, y, z = G.nodes[node]['pos']
        x, y, z = int(x), int(y), int(z)
        x = min(max(x, 0), self._mask.shape[2] - 1)
        y = min(max(y, 0), self._mask.shape[1] - 1)
        z = min(max(z, 0), self._mask.shape[0] - 1)

        return self._mask[z, y, x] > 0

    def _get_ordered_chain(self,
                           G: nx.Graph,
                           nodes: Set[int]) -> Optional[List[int]]:
        """将连通分量转换为有序链"""
        if len(nodes) == 1:
            return [next(iter(nodes))]

        endpoints = [n for n in nodes if G.degree(n) == 1]
        if len(endpoints) != 2:
            raise ValueError("链的端点数量不正确，必须为2个")

        start = endpoints[0]
        visited = set()
        ordered = []
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            ordered.append(node)
            for neighbor in G.neighbors(node):
                if neighbor in nodes and neighbor not in visited:
                    queue.append(neighbor)

        return ordered if len(ordered) == len(nodes) else None

    def _get_tree_and_parent(self, graph, root):
        visited = set()
        queue = [root]
        self.parent[root] = None

        while queue:
            node = queue.pop(0)
            self.in_tree_nodes.append(node)
            if node in visited:
                continue
            visited.add(node)
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in visited and neighbor != self.parent.get(node):
                    self.parent[neighbor] = node
                    queue.append(neighbor)

    def run(self, G, chains, dist_thresh=1, angle_thresh=60, radius_ratio_thresh=1.5):
        # 标记已连接的节点（树中的节点）
        while True:
            # 建立当前树的空间索引
            tree_coords = [G.nodes[n]['pos'] for n in self.in_tree_nodes]
            kdtree = KDTree(tree_coords)

            # 待处理的连接对 (a, b)
            connect_pairs = []

            chain_to_remove = []

            for chain in chains:
                endpoints = [chain[0], chain[-1]]
                if(chain[0] == chain[-1]):
                    endpoints = [chain[0]]
                else:
                    # 只考虑距离soma最近的末梢点
                    distances = {}
                    for ep in endpoints:
                        distances[ep] = np.linalg.norm(np.array(G.nodes[ep]['pos']) - np.array(G.nodes[self.root]['pos']))
                    # 取距离soma最近的末梢点
                    endpoints = [min(endpoints, key=lambda x: distances[x])]

                # ak_ep_num = 0 # 可以连接的末梢点
                for b in endpoints: # a是树中的点，b是链上的点
                    # parent_b_in_chain = self.parent[b]
                    # 如果b有父节点
                    if self.parent.get(b) is not None:
                        continue
                    b_children = [n for n in G.neighbors(b)]

                    # 寻找最近的树节点a
                    b_pos = np.array(G.nodes[b]['pos'])
                    dist, idx = kdtree.query(b_pos)
                    a = list(self.in_tree_nodes)[idx]

                    # 计算树中a到其子节点的方向
                    a_parent = self.parent[a]

                    if(not a_parent): # a是根节点
                        continue

                    if(not b_children): # b是孤立点，直接跳过
                        continue
                    else:
                        b_children = b_children[0]
                        # 计算方向向量（链方向： b -> b_children）
                        vec_chain = np.array(G.nodes[b]['pos']) - np.array(G.nodes[b_children]['pos'])
                        # a_parent -> a
                        vec_tree = np.array(G.nodes[a_parent]['pos']) - np.array(G.nodes[a]['pos'])
                        # 计算夹角
                        cos_theta = np.dot(vec_chain, vec_tree) / (np.linalg.norm(vec_chain) * np.linalg.norm(vec_tree) + 1e-10)
                        # print(cos_theta, np.dot(vec_chain, vec_tree), np.linalg.norm(vec_chain), np.linalg.norm(vec_tree))
                        angle = degrees(acos(np.clip(cos_theta, -1, 1)))
                        valid_angle = (angle < angle_thresh)
                    # 判断是否满足连接条件
                    if (dist < dist_thresh # a和b的距离不能超过阈值
                            and valid_angle # 夹角不能超过阈值
                            and G.nodes[b]['radius'] < radius_ratio_thresh * G.nodes[a]['radius'] # 子节点半径不能大于父节点的radius_ratio_threshold倍
                    ):
                        chain_to_remove.append(chain)
                        connect_pairs.append((a, b))
                        # ak_ep_num = ak_ep_num + 1
                        # chains.remove(chain)
                # if(ak_ep_num == 2):
                #     # 如果两个末梢点都可以连接，直接删除链
                #     connect_pairs = connect_pairs[:-2]
                #     chains.remove(chain)
                # elif(ak_ep_num == 1):
                #     chain_to_remove.append(chain)




            # 如果没有可连接的链，终止
            if not connect_pairs:
                break

            for chain in chain_to_remove:
                if(chain in chains):
                    chains.remove(chain)

            # 处理所有待连接对
            for a, b in connect_pairs:
                # 添加连接边
                G.add_edge(a, b)
                self.parent[b] = a
                self.in_tree_nodes.append(b)

                # 将链中其他节点的父子关系加入
                current = b
                while True:
                    # 找到链中current的父节点（未在树中的邻居）
                    next_nodes = [n for n in G.neighbors(current)
                                  if n not in self.in_tree_nodes]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    self.parent[next_node] = current
                    self.in_tree_nodes.append(next_node)
                    current = next_node

    # 合并几乎重合的点
    def _merge_nearby_nodes(self, G, threshold=0.1):
        # bfs
        start_node = self.in_tree_nodes[0]
        visited = set()
        queue = deque([self.root])
        while queue:
            u = queue.popleft()  # Current parent node

            # Iterate over all children of u (neighbors except parent)
            for v in list(G.neighbors(u)):  # Use list() to avoid dynamic changes
                if v in visited:
                    continue  # Skip already visited (parent) nodes

                # Check distance between u and v
                dist = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
                if dist < threshold:
                    # --- Merge v into u ---
                    # 1. Transfer v's children to u
                    for child in list(G.neighbors(v)):
                        if child != u:  # Avoid creating self-loops
                            G.add_edge(u, child)

                    # 2. Update u's position and radius (optional)
                    # G.nodes[u]['pos'] = (G.nodes[u]['pos'] + G.nodes[v]['pos']) / 2
                    a_pos, b_pos = G.nodes[u]['pos'], G.nodes[v]['pos']
                    G.nodes[u]['pos'] = (a_pos[0] + b_pos[0]) / 2, (a_pos[1] + b_pos[1]) / 2, (a_pos[2] + b_pos[2]) / 2
                    G.nodes[u]['radius'] = (G.nodes[u]['radius'] + G.nodes[v]['radius']) / 2

                    # 3. Remove v
                    G.remove_node(v)
                else:
                    # If not merged, add v to the queue for BFS
                    queue.append(v)
                    visited.add(v)

    # 删除完全在mask中的链
    def _remove_chain_in_mask(self, G):
        # 获取所有叶子结点
        leaf_nodes = [node for node in G.nodes() if G.degree(node) == 1]
        for leaf_node in leaf_nodes:
            # path to root
            if(not leaf_node in self.in_tree_nodes):
                continue
            path_to_root = nx.shortest_path(G, source=leaf_node, target=self.root)
            # 如果路径上的所有节点都在mask中，则删除该链
            if all(self._check_node_in_mask(G, node) for node in path_to_root):
                # 删除链
                if(self.root in path_to_root):
                    path_to_root.remove(self.root)
                G.remove_nodes_from(path_to_root)


def retrace(swc_file, seg_file, soma_file, output_file):
    if(os.path.exists(output_file)):
        return False
    if(not os.path.exists(swc_file) or not os.path.exists(seg_file) or not os.path.exists(soma_file)):
        return False
    neuron_id = int(os.path.basename(seg_file).split("_")[1].split(".")[0])
    soma = expand_soma_to_origin_size(seg_file, soma_file)
    xy_resolution, z_resolution = meta_info[meta_info['cell_id'] == neuron_id][
        ['xy_resolution', 'z_resolution']].values[0]
    resolution = (float(xy_resolution)/1000, float(xy_resolution)/1000, float(z_resolution)/1000)
    # print(soma.shape)
    soma = np.flip(soma, axis=1)
    resized_size = (int(soma.shape[0]*resolution[2]), int(soma.shape[1]*resolution[1]), int(soma.shape[2]*resolution[0]))
    soma = resize(soma, resized_size, order=0, mode='edge', anti_aliasing=False)


    swc_g = load_swc_to_undirected_graph(swc_file, resolution=resolution)

    # 第一步，在所有分支点进行断连
    swc_g = disconnect_with_shadow_nodes(swc_g)

    # 第二步，删除mask中完全包含的链
    mask = soma > 0
    chainprocessor = GenerateSTEMChain()
    swc_g, root_node = chainprocessor.process(swc_g, mask)

    # 第三步，把链加回去
    chainadder = ChainAdder(swc_g, root_node, mask, 5, 90, 1.5)

    export_to_swc_dfs(swc_g, root_node,output_file, resolution)

def try_retrace(swc_file, seg_file, soma_file, output_file):
    if(os.path.exists(output_file)):
        return False
    if(not os.path.exists(swc_file) or not os.path.exists(seg_file) or not os.path.exists(soma_file)):
        return False
    try:
        retrace(swc_file, seg_file, soma_file, output_file)
    except Exception as e:
        print(f"Error processing {swc_file}: {e}")
        return False
    return True

if __name__ == '__main__':
    ws_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k"
    seg_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_seg"
    soma_seg_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_soma_seg"
    swc_dir = os.path.join(ws_dir, "down_sampled_swcs")
    output_dir = os.path.join(ws_dir, "down_sampled_retrace_swcs")
    os.makedirs(output_dir, exist_ok=True)

    # swc_file = os.path.join(swc_dir, "image_15199.swc")
    # seg_file = os.path.join(seg_dir, "image_15199.tif")
    # soma_file = os.path.join(soma_seg_dir, "image_15199.tif")
    # output_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/image_15199_retrace.swc"

    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.tif')]
    Parallel(n_jobs=8)(
        delayed(try_retrace)(
            os.path.join(swc_dir, f.replace(".tif", ".swc")),
            os.path.join(seg_dir, f),
            os.path.join(soma_seg_dir, f),
            os.path.join(output_dir, f.replace(".tif", ".swc"))
        ) for f in tqdm(seg_files)
    )





