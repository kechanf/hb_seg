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
import sys
import io
from contextlib import contextmanager
import platform
import subprocess
import tifffile
from tempfile import TemporaryDirectory
# import dilate
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

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

def get_soma_around_e_cut(swc_file=r"C:\Users\12626\Desktop\50k\swc\15533.swc", ptrs_dir=r"C:\Users\12626\Desktop\50k\swc\PTRSB_Somas", save_swc_file=None):
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
                               header=None, names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g', 'color_b'])
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
    e = ECut(tree, tree_nodenum)  # 0 and 100 are the IDs of somata
    e.run()
    trees = e.export_swc()
    for i in range(len(potential_somanum_in_swc)):
        write_swc(trees[potential_somanum_in_swc[i][1]], save_swc_file.replace('.swc', f'_{potential_somanum_in_swc[i][0]}.swc'))

def swc2img(swc_file, img_shape, xy_resolution, out_file=None, v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i <input.swc> [-p <sz0> <sz1> <sz2>] [-o <output_image.raw>]\n"
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_filter -i <input.tif> <input.swc> [-o <output_image.raw>]\n"
    swc_point_list = pd.read_csv(swc_file, sep=' ', comment='#', header=None,
                                 names=['n', 'type', 'x', 'y', 'z', 'r', 'pn'])
    swc_point_list['x'] = swc_point_list['x'] / (xy_resolution / 1000)
    swc_point_list['y'] = swc_point_list['y'] / (xy_resolution / 1000)
    with TemporaryDirectory() as temp_dir:
        temp_swc_file = os.path.join(temp_dir, os.path.basename(swc_file))
        swc_point_list.to_csv(temp_swc_file, sep=' ', header=False, index=False)
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i {temp_swc_file} ' \
                  f'-p {img_shape[2]} {img_shape[1]} {img_shape[0]} -o {out_file}'
        cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
        print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)

tif_file = "/data2/kfchen/tracing_ws/muti_channel_test/muti_channel_img_3.tif"
img = tifffile.imread(tif_file)
# mip = np.max(img, axis=0)
# mip = np.max(mip, axis=0)
# tifffile.imsave("/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/muti_channel_img_mip_full.png", mip)
#
colors = [
    (0, 0, 1),  # 蓝色 (通道 2)
    (0, 1, 0),  # 绿色 (通道 1)
    (1, 0.5, 0),  # 橙色 (通道 3)
    (1, 0, 1),  # 紫色 (通道 4)
]

# 计算每个通道的 MIP
mips = [np.max(img[:, :, :, i], axis=0).astype(float) for i in range(4)]
mips = [mip / np.max(mip) for mip in mips]
mip_rgb = np.zeros((mips[0].shape[0], mips[0].shape[1], 3))  # 初始化 RGB 图像

# 将每个通道的 MIP 叠加到 RGB 图像中
for i, (mip, color) in enumerate(zip(mips, colors)):
    for c in range(3):  # 遍历 RGB 通道
        mip_rgb[:, :, c] += mip * color[c]  # 将 MIP 按颜色权重叠加

# 归一化到 [0, 1] 范围
# mip_rgb = np.clip(mip_rgb, 0, 1)

# 显示结果
# plt.figure(figsize=(8, 8))
plt.imshow(mip_rgb)
# plt.title("MIP of 4-Channel 3D Image")
# plt.axis('off')
plt.savefig("/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/muti_channel_img_mip_new.png")
plt.close()

origin_img_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/image/image_15533_0000.tif"
origin_img = tifffile.imread(origin_img_file)
plt.imshow(np.max(origin_img, axis=0))
plt.savefig("/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/muti_channel_img_mip_origin.png")
plt.close()
exit()

# print(img.shape)
# color_list = ["Blues", "Greens", "Oranges", "Purples"]
# for channel in range(img.shape[0]):
#     # tifffile.imsave(f"/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/muti_channel_img_{channel}.png", np.max(img[channel], axis=0))
#     current_mask = np.max(img[channel], axis=0)
#     plt.imshow(current_mask, cmap=color_list[channel])
#     plt.savefig(f"/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/muti_channel_img_{channel}.png")
#     plt.close()

# show_muti_mip = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
# color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
# for channel in range(img.shape[0]):
#     current_mask = np.max(img[channel], axis=0)
#     show_muti_mip[current_mask > 0] = color_list[channel]
# # tifffile.imsave(f"/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/muti_channel_img_mip.png", show_muti_mip)
# plt.imshow(show_muti_mip)
# plt.savefig(f"/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/muti_channel_img_mip.png")


# exit()


test_id = 15533
swc_file = f"/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/8_estimated_radius_swc/{test_id}.swc"
ptrs_dir = r"/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas"
save_swc_file = f"/data2/kfchen/tracing_ws/muti_channel_test/{test_id}.swc"
# get_soma_around_e_cut(swc_file, ptrs_dir, save_swc_file)

img_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/image/image_15533_0000.tif"
img = tifffile.imread(img_file)
xy_resolution = meta_info[meta_info['cell_id'] == 15533]['xy_resolution'].values[0]
# for swc_file in os.listdir("/data2/kfchen/tracing_ws/muti_channel_test"):
#     if(not swc_file.endswith('.swc')):
#         continue
#     swc_file = os.path.join("/data2/kfchen/tracing_ws/muti_channel_test", swc_file)
#     swc2img(swc_file, img.shape, xy_resolution, out_file=swc_file.replace('.swc', '.tif'))

muti_channel_num = 4
for iterations in range(3, 4):
    muti_channel_img = np.zeros((img.shape[0], img.shape[1], img.shape[2], muti_channel_num), dtype=np.uint8)
    for img_mask_file in tqdm(os.listdir("/data2/kfchen/tracing_ws/muti_channel_test")):
        if(not img_mask_file.endswith('.tif')):
            continue
        if('muti_channel_img' in img_mask_file):
            continue
        img_mask = tifffile.imread(os.path.join("/data2/kfchen/tracing_ws/muti_channel_test", img_mask_file))
        dilated_img_mask =  binary_dilation(img_mask, structure=np.ones((3,3,3)), iterations=iterations)

        current_img = img[dilated_img_mask > 0]
        # copy to random channel
        # muti_channel_img[np.random.randint(0, muti_channel_num), dilated_img_mask > 0] = current_img
        muti_channel_img[dilated_img_mask > 0, np.random.randint(0, muti_channel_num)] = current_img

    # save muti channel image
    muti_channel_img_file = f"/data2/kfchen/tracing_ws/muti_channel_test/muti_channel_img_{iterations}.tif"
    muti_channel_img = muti_channel_img.astype("uint8")
    tifffile.imsave(muti_channel_img_file, muti_channel_img)







