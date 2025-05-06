import pandas as pd
import networkx as nx
import numpy as np
import tifffile
from joblib import Parallel, delayed
from tqdm import tqdm
import os

class SWC_Radius_Estimator:
    '''
    Usage:
    swc_file = 'example.swc'
    img_file = 'example.tif'
    output_swc_file = 'example_radius_estimated.swc'
    swc_radius_estimator = SWC_Radius_Estimator(swc_file, img_file, output_swc_file)

    '''
    def __init__(self, swc_file, img_file,
                 output_swc_file=None,
                 background_tolerance_ratio=0.05, flip_y=False, max_r=30, min_r=1, search_precision=0.25, smooth_sigma=1.0):
        self.swc_file = swc_file
        self.img_file = img_file
        self.background_tolerance_ratio = background_tolerance_ratio
        self.flip_y = flip_y
        self.max_r = max_r
        self.min_r = min_r
        self.search_precision = search_precision
        self.output_swc_file = output_swc_file

        self.swc_tree = self.generate_tree_from_swc_file()
        self.img = self.load_img()

        self.estimate_radius()
        # smooth
        self.swc_tree = self.gaussian_smoothing_radius_tree(self.swc_tree, sigma=smooth_sigma)
        self.save_swc()

    def load_swc(self):
        swc = pd.read_csv(self.swc_file, sep=' ', header=None, comment='#')
        swc = swc.iloc[:, :7]
        swc.columns = ['n', 'type', 'x', 'y', 'z', 'r', 'parent']
        return swc

    def load_img(self):
        img = tifffile.imread(self.img_file)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        if self.flip_y:
            img = np.flip(img, axis=1)
        return img

    def generate_tree_from_swc_file(self):
        swc = self.load_swc()
        G = nx.DiGraph()
        for i in range(swc.shape[0]):
            n = swc.iloc[i]['n']
            type = swc.iloc[i]['type']
            x = swc.iloc[i]['x']
            y = swc.iloc[i]['y']
            z = swc.iloc[i]['z']
            r = swc.iloc[i]['r']
            parent = swc.iloc[i]['parent']
            G.add_node(n, x=x, y=y, z=z, r=r, type=type)
            if parent != -1:
                G.add_edge(parent, n)
        return G

    def estimate_radius(self):
        node_list = list(self.swc_tree.nodes())
        Parallel(n_jobs=8, backend="threading")(delayed(self.estimate_radius_at_node)(node) for node in node_list)
        # for node in node_list:
        #     self.estimate_radius_at_node(node)
        # print(self.swc_tree.nodes[1]['r'])
        # return self.swc_tree

    def is_signal_in_sphere(self, distances, node_voxel_value, radius, foucs_img=None):
        sphere_mask = distances <= radius
        sphere_values = foucs_img[sphere_mask]
        if sphere_values.size > 0:
            mean_intensity = sphere_values.mean()
        else:
            mean_intensity = 0
        # if(mean_intensity>0):
        #     print(mean_intensity, node_voxel_value)

        if (mean_intensity < node_voxel_value * (1 - self.background_tolerance_ratio)
                or mean_intensity > node_voxel_value * (1 + self.background_tolerance_ratio))\
                or mean_intensity < 0.2:
            return False
        else:
            return True

    def estimate_radius_at_node(self, node):
        node_x = int(self.swc_tree.nodes[node]['x'])
        node_y = int(self.swc_tree.nodes[node]['y'])
        node_z = int(self.swc_tree.nodes[node]['z'])
        max_r, min_r = self.max_r, self.min_r

        if(node_x < 0 or node_x >= self.img.shape[2] or node_y < 0 or node_y >= self.img.shape[1] or node_z < 0 or node_z >= self.img.shape[0]):
            self.swc_tree.nodes[node]['r'] = min_r
            return

        node_voxel_value = self.img[node_z, node_y, node_x]
        if(node_voxel_value == 0):
            self.swc_tree.nodes[node]['r'] = min_r
            return


        z_start, z_end, y_start, y_end, x_start, x_end = node_z - max_r, node_z + max_r, node_y - max_r, node_y + max_r, node_x - max_r, node_x + max_r
        z_start, z_end = max(0, z_start), min(self.img.shape[0], z_end)
        y_start, y_end = max(0, y_start), min(self.img.shape[1], y_end)
        x_start, x_end = max(0, x_start), min(self.img.shape[2], x_end)
        foucs_img = self.img[z_start:z_end, y_start:y_end, x_start:x_end]
        z, y, x = np.meshgrid(np.arange(z_start, z_end), np.arange(y_start, y_end), np.arange(x_start, x_end), indexing='ij')
        distances = np.sqrt((x - node_x) ** 2 + (y - node_y) ** 2 + (z - node_z) ** 2)

        mar_r, min_r = float(max_r), float(min_r)
        # 二分
        while max_r - min_r > self.search_precision:
            r = (max_r + min_r) / 2
            if self.is_signal_in_sphere(distances, node_voxel_value, r, foucs_img):
                min_r = r
            else:
                max_r = r

        self.swc_tree.nodes[node]['r'] = min_r
        # print(f'Node {node} radius: {self.swc_tree.nodes[node]["r"]}')

    def gaussian_smoothing_radius_tree(self, G, sigma=1.0):
        def calc_node_dist(G, node1, node2):
            pos1 = np.array([G.nodes[node1]['x'], G.nodes[node1]['y'], G.nodes[node1]['z']])
            pos2 = np.array([G.nodes[node2]['x'], G.nodes[node2]['y'], G.nodes[node2]['z']])
            return np.linalg.norm(pos1 - pos2)
        smoothed_values = {}
        soma_r = G.nodes[1]['r']
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            weights = []
            values = []
            for neighbor in neighbors:
                distance = calc_node_dist(G, node, neighbor)
                weight = np.exp(- (distance ** 2) / (2 * sigma ** 2))
                weights.append(weight)
                values.append(G.nodes[neighbor]['r'])
            # 自身的权重
            self_weight = np.exp(0)
            total_weight = self_weight + sum(weights)
            weighted_sum = G.nodes[node]['r'] * self_weight + sum(w * v for w, v in zip(weights, values))
            smoothed_values[node] = weighted_sum / total_weight
        nx.set_node_attributes(G, smoothed_values, 'r')
        G.nodes[1]['r'] = soma_r
        return G

    def save_swc(self):
        if(self.output_swc_file is None):
            self.output_swc_file = self.swc_file.replace('.swc', '_radius_estimated.swc')
        with open(self.output_swc_file, 'w') as f:
            f.write('# Generated by SWC_Radius_Estimator\n')
            for node in self.swc_tree.nodes():
                type = self.swc_tree.nodes[node]['type']
                x = self.swc_tree.nodes[node]['x']
                y = self.swc_tree.nodes[node]['y']
                z = self.swc_tree.nodes[node]['z']
                r = self.swc_tree.nodes[node]['r']
                # print(f"!!Node {node} radius: {r}")
                parent = list(self.swc_tree.predecessors(node))
                if len(parent) == 0:
                    parent = -1
                else:
                    parent = parent[0]
                f.write(f'{int(node)} {int(type)} {x} {y} {z} {r} {int(parent)}\n')
        if("_radius_estimated.swc" in self.output_swc_file):
            os.remove(self.swc_file)
            os.rename(self.output_swc_file, self.swc_file)




