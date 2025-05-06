import os
import time
from simple_swc_tool.plot_neuron import plot_img_on_fig, plot_swc_on_fig, plot_markers_on_fig
from simple_swc_tool.convert_eswc_to_swc import eswc2swc
import pandas as pd
from v3dpy.loaders import Raw, PBD
import tifffile as tiff
import time
import numpy as np
import shutil
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt

meta_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/meta_info_0205.xlsx"
meta = pd.read_excel(meta_file)

def get_soma_markers(PTRSB_dir, DB_id):
    PTRSB_files = [f for f in os.listdir(PTRSB_dir) if f.startswith(DB_id) and f.endswith('.marker')]
    if(not len(PTRSB_files) == 1):
        return None
    soma_markers_list = []
    for PTRSB_file in PTRSB_files:
        PTRSB_file_path = os.path.join(PTRSB_dir, PTRSB_file)
        soma_markers = pd.read_csv(PTRSB_file_path, sep=',',
                                   comment='#',
                                   names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
                                          'color_b'])
        soma_markers_list.append(soma_markers)
    soma_markers = pd.concat(soma_markers_list)

    return soma_markers

def plot(img_file, swc_file, mip_file):

    neuron_id = int(os.path.basename(img_file).split('_')[0].split(".")[0])
    if (os.path.exists(mip_file)):
        return

    img = tiff.imread(img_file)
    # 三视图
    img_mip = np.max(img, axis=0)
    bkg_shape = (int(img_mip.shape[0]), int(img_mip.shape[1]))
    img_mip = resize(img_mip, bkg_shape)

    background = np.ones(bkg_shape).astype(np.uint8) * 255
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

    background = plot_img_on_fig(background, img, projection_direction="xy")
    background = plot_swc_on_fig(background, swc_file, line_color=(255, 0, 0), projection_direction="xy", plot_mode="line")
    # save
    tiff.imwrite(mip_file, background)


def prepare_swc():
    eswc_in_neuron_block_dir = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_neuron_block/eswc"
    swc_in_neuron_block_dir =  "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_neuron_block/swc" # 坐标是全局坐标
    old_new_id_map_file = "/data/kfchen/trace_ws/paper_trace_result/id_map.xlsx"
    id_map_df = pd.read_excel(old_new_id_map_file)[["cell_id_backUp", "cell_id"]]
    id_map_df = id_map_df.rename(columns={"cell_id_backUp": "old_id", "cell_id": "new_id"})
    id_map = {}
    for i in range(len(id_map_df)):
        id_map[id_map_df.iloc[i, 0]] = id_map_df.iloc[i, 1] # old -> new

    os.makedirs(swc_in_neuron_block_dir, exist_ok=True)
    eswc_files = [f for f in os.listdir(eswc_in_neuron_block_dir) if f.endswith('.eswc')]
    for eswc_file in eswc_files:
        eswc_file_path = os.path.join(eswc_in_neuron_block_dir, eswc_file)
        old_id = int(eswc_file.split('_')[0])
        if(not old_id in id_map):
            continue
        new_id = id_map[old_id]
        swc_file_path = os.path.join(swc_in_neuron_block_dir, f"{new_id}.swc")
        if(not os.path.exists(swc_file_path)):
            eswc2swc(eswc_file_path, swc_file_path)
    swc_in_neuron_block_with_radius_dir = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_neuron_block/swc_with_radius"
    os.makedirs(swc_in_neuron_block_with_radius_dir, exist_ok=True)


# temp_img_dir = os.path.join(radius_estimate_ws, 'temp_img')
# temp_swc_dir = os.path.join(radius_estimate_ws, 'temp_swc')
# os.makedirs(temp_img_dir, exist_ok=True)
# os.makedirs(temp_swc_dir, exist_ok=True)
# for swc_file in os.listdir(swc_in_neuron_block_dir):
#     swc_file_path = os.path.join(swc_in_neuron_block_dir, swc_file)
#     img_id = int(swc_file.split('_')[0])
#     img_path = id_img_map[img_id]
#     # pbd = PBD()
#     # img = pbd.load(img_path)[0]
#     raw = Raw()
#     img = raw.load(img_path)[0]
#
#     xy_resolution = meta[meta['cell_id'] == img_id]['xy_resolution'].values[0]
#
#     shutil.copy(swc_file_path, os.path.join(temp_swc_dir, f"{img_id}.swc"))
#     tiff.imwrite(os.path.join(temp_img_dir, f"{img_id}.tif"), img)
#     exit()


def estimate_radius():
    swc_in_neuron_block_dir =  "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_neuron_block/swc" # 坐标是全局坐标
    swc_in_DB_root = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_DB"

    DB_neuron_map = {}
    swc_in_neuron_block_files = [f for f in os.listdir(swc_in_neuron_block_dir) if f.endswith('.swc')]
    for swc_file in swc_in_neuron_block_files:
        swc_file_path = os.path.join(swc_in_neuron_block_dir, swc_file)
        PTRSB = meta[meta['cell_id'] == int(swc_file.split('.')[0])]['PTRSB']
        if(len(PTRSB) == 0):
            continue
        PTRSB = PTRSB.values[0]
        if(PTRSB not in DB_neuron_map):
            DB_neuron_map[PTRSB] = []
        DB_neuron_map[PTRSB].append(swc_file_path)


    PTRSB_soma_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas"
    # DB_img_root = "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/PTRSB_DB"
    # DB_img_files = []
    # # walk
    # for root, dirs, files in os.walk(DB_img_root):
    #     for file in files:
    #         if file.endswith('_8bit.v3draw'):
    #             DB_img_files.append(os.path.join(root, file))
    v3d_img_root = "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/Cell_Images"
    id_img_map = {}
    # walk
    for root, dirs, files in os.walk(v3d_img_root):
        for file in files:
            if file.endswith('.v3dpbd'):
                id_img_map[int(file.split('.')[0])] = os.path.join(root, file)


    soma_block_size = (100, 100, 100) # xyz
    radius_estimate_ws = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_neuron_block/radius_estimate_ws"
    mip_dir = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_neuron_block/radius_estimate_ws/mip"
    for DB_id in DB_neuron_map:
        # print(DB_id)
        DB_neuron_dir = os.path.join(swc_in_DB_root, DB_id)
        os.makedirs(DB_neuron_dir, exist_ok=True)
        # print(len(DB_neuron_map[DB_id]))

        soma_markers = get_soma_markers(PTRSB_soma_dir, DB_id)
        if(soma_markers is None):
            continue
        # print(soma_markers.shape)
        print(f"DB_id: {DB_id}, soma_markers: {soma_markers.shape[0]}, reconstructions: {len(DB_neuron_map[DB_id])}")

        # DB_img_file = [f for f in DB_img_files if DB_id in f]
        # if(not len(DB_img_file) == 1):
        #     print(f"DB_id: {DB_id}, DB_img_file: {DB_img_file}")
        #     print("fuck")
        #     continue
        # DB_img_file = DB_img_file[0]
        # print("start load", DB_img_file)
        # start_time = time.time()
        # raw = Raw()
        # img = raw.load(DB_img_file)[0]
        # print("load ok, time cost is ", time.time() - start_time)

        for swc_file_path in DB_neuron_map[DB_id]:
            cell_id = int(os.path.basename(swc_file_path).split('_')[0].split(".")[0])
            print(cell_id, )
            global_soma_pos = soma_markers[soma_markers['name'] == cell_id][['x', 'y', 'z']].values[0]
            soma_pos_in_neuron_block = meta[meta['cell_id'] == cell_id][["soma_x", "soma_y", "soma_z"]] # 在切好的neuron块中的soma位置
            # to tuple
            soma_pos_in_neuron_block = tuple(soma_pos_in_neuron_block.iloc[0])
            soma_pos_in_neuron_block = (float(soma_pos_in_neuron_block[0]), float(soma_pos_in_neuron_block[1]), float(soma_pos_in_neuron_block[2]))
            xy_resolution = float(meta[meta['cell_id'] == cell_id]['xy_resolution'].values[0])

            # 准备soma block img
            current_soma_block_size = (soma_block_size[0] * 1000/ xy_resolution, soma_block_size[1] * 1000 / xy_resolution, soma_block_size[2]) # xyz
            print(soma_pos_in_neuron_block, current_soma_block_size)
            neuron_img_file = id_img_map[cell_id]
            # raw = Raw()
            # img = raw.load(neuron_img_file)[0]
            pbd = PBD()
            img = pbd.load(neuron_img_file)

            print(img.shape)
            img = img[0]
            img = np.flip(img, axis=1)


            # soma_block_x_start, soma_block_x_end = soma_pos_in_neuron_block[0] - current_soma_block_size[0] / 2, soma_pos_in_neuron_block[0] + current_soma_block_size[0] / 2
            # soma_block_y_start, soma_block_y_end = soma_pos_in_neuron_block[1] - current_soma_block_size[1] / 2, soma_pos_in_neuron_block[1] + current_soma_block_size[1] / 2
            # soma_block_z_start, soma_block_z_end = soma_pos_in_neuron_block[2] - current_soma_block_size[2] / 2, soma_pos_in_neuron_block[2] + current_soma_block_size[2] / 2
            # print(soma_block_x_start, soma_block_x_end, soma_block_y_start, soma_block_y_end, soma_block_z_start, soma_block_z_end)
            #
            # soma_block_x_start, soma_block_x_end = max(0, soma_block_x_start), min(img.shape[2], soma_block_x_end)
            # soma_block_y_start, soma_block_y_end = max(0, soma_block_y_start), min(img.shape[1], soma_block_y_end)
            # soma_block_z_start, soma_block_z_end = max(0, soma_block_z_start), min(img.shape[0], soma_block_z_end)
            # print(soma_block_x_start, soma_block_x_end, soma_block_y_start, soma_block_y_end, soma_block_z_start, soma_block_z_end)
            #
            # soma_block = img[int(soma_block_z_start):int(soma_block_z_end), int(soma_block_y_start):int(soma_block_y_end), int(soma_block_x_start):int(soma_block_x_end)]
            soma_block = img
            tiff.imwrite(os.path.join(radius_estimate_ws, f"{cell_id}.tif"), soma_block)
            #
            # # 准备soma block swc
            # soma_pos_in_target_neuron_block = (current_soma_block_size[0] / 2, current_soma_block_size[1] / 2, current_soma_block_size[2] / 2)
            swc = pd.read_csv(swc_file_path, sep=' ', comment='#', header=None,
                              names=['n', 'type', 'x', 'y', 'z', 'r', 'parent'])
            # offset = (global_soma_pos[0] - current_soma_block_size[0] / 2, global_soma_pos[1] - current_soma_block_size[1] / 2, global_soma_pos[2] - current_soma_block_size[2] / 2)
            swc_soma_pos = swc[swc['parent'] == -1][['x', 'y', 'z']].values[0]
            offset = (soma_pos_in_neuron_block[0] - swc_soma_pos[0], soma_pos_in_neuron_block[1] - swc_soma_pos[1], soma_pos_in_neuron_block[2] - swc_soma_pos[2])
            swc['x'] = swc['x'] + offset[0]
            swc['y'] = swc['y'] + offset[1]
            swc['z'] = swc['z'] + offset[2]
            # save
            swc.to_csv(os.path.join(radius_estimate_ws, f"{cell_id}.swc"), sep=' ', header=False, index=False)
            #
            # print(f"cell_id: {cell_id}, global_soma_pos: {global_soma_pos}, xy_resolution: {xy_resolution}, soma_block_size: {soma_block_size}, current_soma_block_size: {current_soma_block_size}")
            mip_file = os.path.join(mip_dir, f"{cell_id}.tif")
            plot(os.path.join(radius_estimate_ws, f"{cell_id}.tif"), os.path.join(radius_estimate_ws, f"{cell_id}.swc"), mip_file)
            # exit()

prepare_swc()
estimate_radius()




