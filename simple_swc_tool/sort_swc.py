import os.path

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
import matplotlib.pyplot as plt
import itertools


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


def apply_clustering(G, eps=1, min_samples=1):
    # 提取位置信息并应用DBSCAN聚类
    positions = np.array([G.nodes[n]['pos'] for n in G.nodes()])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    # 更新节点信息
    for node_id, cluster_id in zip(G.nodes, clustering.labels_):
        G.nodes[node_id]['cluster'] = cluster_id
    return clustering.labels_


def merge_clusters(G, labels):
    # 构建每个聚类的节点列表
    cluster_groups = {}
    for node_id, cluster_id in zip(G.nodes, labels):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(node_id)

    new_G = nx.Graph()
    for cluster_id, nodes in cluster_groups.items():
        # 计算合并后的属性
        if cluster_id == -1:  # 跳过噪声点
            continue
        x, y, z, radius = zip(*[(G.nodes[n]['pos'][0], G.nodes[n]['pos'][1], G.nodes[n]['pos'][2], G.nodes[n]['radius']) for n in nodes])
        avg_pos = (np.mean(x), np.mean(y), np.mean(z))
        avg_radius = np.mean(radius)
        new_G.add_node(cluster_id, pos=avg_pos, radius=avg_radius)

    # 更新边，连接聚类中心
    for u, v, data in G.edges(data=True):
        cluster_u = G.nodes[u]['cluster']
        cluster_v = G.nodes[v]['cluster']
        if cluster_u != cluster_v and cluster_u != -1 and cluster_v != -1:
            new_G.add_edge(cluster_u, cluster_v)

    return new_G


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

def check_connectivity(G):
    # 对于无向图，直接检查连通性
    if nx.is_connected(G):
        # print("Graph is connected")
        return True
    else:
        connected_components = list(nx.connected_components(G))
        # 计算连通块的数量
        num_connected_components = len(connected_components)
        # print(f"Graph is not connected; it has {num_connected_components} connected components")
        # print("Graph is not connected")
        return False


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
    start_node = find_nearest_node(G, root_pos)

    # 调整根节点
    potential_root = max(G.nodes, key=lambda x: G.degree(x))
    potential_root_degree = G.degree(potential_root)
    potential_root_list = [node for node in G.nodes if G.degree(node) == potential_root_degree]
    for node in potential_root_list:
        if G.degree(node) > 4 and len(potential_root_list) == 1: # 这个点的度数大于4
            start_node = node
        elif(nx.shortest_path_length(G, start_node, node) < 3):
            start_node = node
        elif(np.linalg.norm(np.array(G.nodes[node]['pos']) - np.array(root_pos)) < 10):
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
                if(parent_id == -1):
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


def transform_to_trees(G):
    # 新建一个图来存储结果
    tree_G = nx.Graph()

    # 获取G的所有连通分量
    components = nx.connected_components(G)

    # 对每个连通分量计算最小生成树
    for component in components:
        # 提取子图
        subgraph = G.subgraph(component)

        # 计算最小生成树
        if nx.is_connected(subgraph):
            mst = nx.minimum_spanning_tree(subgraph)
            # 添加最小生成树到结果图中
            tree_G = nx.compose(tree_G, mst)
        else:
            raise ValueError("Subgraph is not connected, which should not happen.")

    return tree_G

def connect_components(G):
    # 如果图已经是连通的，直接返回
    if nx.is_connected(G):
        return G

    # 新建图，包含所有原始图的节点和边
    new_G = G.copy()

    # 获得连通组件
    components = list(nx.connected_components(G))
    component_list = [comp for comp in components]

    while len(component_list) > 1:
        min_distance = float('inf')
        best_pair = None

        # 遍历所有组件对，寻找最短边
        for i in range(len(component_list)):
            for j in range(i + 1, len(component_list)):
                for u in component_list[i]:
                    for v in component_list[j]:
                        # 这里的距离计算应根据实际情况进行调整
                        # 假设节点u和v的位置在节点属性'pos'中
                        if 'pos' in G.nodes[u] and 'pos' in G.nodes[v]:
                            distance = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
                            if distance < min_distance:
                                min_distance = distance
                                best_pair = (u, v)

        # 添加最短边
        if best_pair:
            new_G.add_edge(best_pair[0], best_pair[1], weight=min_distance)
            # 更新连通组件
            component_list = list(nx.connected_components(new_G))

    return new_G


def sort_swc(swc_file, sorted_swc_file, root_pos=(0,0,0), merge_nodes=True):
    G = load_swc_to_undirected_graph(swc_file)

    if(merge_nodes):
        # 临近点聚类
        labels = apply_clustering(G)
        G = merge_clusters(G, labels)
        # visualize_graph(new_G)


    if(not check_connectivity(G) or not is_tree(G)):
        # 每个连通块生成最小生成树
        G = transform_to_trees(G)
        # 连接各个连通块
        G = connect_components(G)

    # export_to_swc(new_G, '/data/kfchen/trace_ws/to_gu/origin_swc/2364_clustered.swc')
    if(os.path.exists(sorted_swc_file)):
        os.remove(sorted_swc_file)
    # export_to_swc_dfs(G, actual_root, sorted_swc_file)
    export_to_swc_dfs(G, root_pos, sorted_swc_file)

if __name__ == '__main__':
    swc_file = '/data/kfchen/trace_ws/to_gu/origin_swc/3239.swc'
    sorted_swc_file = '/data/kfchen/trace_ws/to_gu/origin_swc/3239_clustered.swc'
    sort_swc(swc_file, sorted_swc_file, root_pos=(0,0,0))
