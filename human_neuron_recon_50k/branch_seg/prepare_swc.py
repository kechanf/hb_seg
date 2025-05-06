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
from simple_swc_tool.swc2tif import swc2img
from tqdm import tqdm
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

meta_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/meta_info_0205.xlsx"
meta_info = pd.read_excel(meta_file)

eswc_in_DB = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_DB/eswc"
swc_in_DB = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_DB/swc" # 坐标是全局坐标
temp_branch_mask_dir = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_DB/temp_branch_mask"
temp_mip_dir = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_DB/temp_mip"
temp_soma_block_dir = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_DB/temp_soma_block"
temp_soma_seg_dir = "/data2/kfchen/tracing_ws/branch_seg/dense_anno_in_DB/temp_soma_seg"
os.makedirs(swc_in_DB, exist_ok=True)
PTRSB_soma_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas"


if __name__ == "__main__":
    eswc_files = [f for f in os.listdir(eswc_in_DB) if f.endswith(".eswc")]

    # for eswc_file in eswc_files:
    #     eswc_id = int(eswc_file.split("_")[0])
    #     swc_file = f"{eswc_id}.swc"
    #     eswc2swc(os.path.join(eswc_in_DB, eswc_file), os.path.join(swc_in_DB, swc_file))

    soma_predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device('cuda', 0),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
    # /data2/kfchen/nnUNet/nnUNet_results/Dataset191_full_res/nnUNetTrainer__nnUNetPlans__3d_fullres
    soma_predictor.initialize_from_trained_model_folder(
        "/data2/kfchen/nnUNet/nnUNet_results/Dataset201_hb_soma/nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    # img_file = "/data2/kfchen/tracing_ws/soma_seg/tif_image/P00121-T001-R001-S005-B1_0000.tif"
    # soma_markers_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas/P00121-T001-R001-S005-B1.apo"
    img_file = "/data2/kfchen/tracing_ws/soma_seg/tif_image/P00122-T001-R001-S006-B1_0000.tif"
    soma_markers_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas/P00122-T001-R001-S006-B1.apo"
    img = tiff.imread(img_file)
    soma_block_size = (256, 256, 256)
    # 更改半径
    swc_files = [f for f in os.listdir(swc_in_DB) if f.endswith(".swc")]
    for swc_file in tqdm(swc_files):
        swc = pd.read_csv(os.path.join(swc_in_DB, swc_file), sep=" ", header=None, comment="#")
        origin_swc = swc.copy()
        swc[5] = 2
        min_x, min_y, min_z = swc[[2, 3, 4]].min()
        min_x, min_y, min_z = int(min_x-1), int(min_y-1), int(min_z-1)
        x_start, y_start, z_start = min_x, min_y, min_z
        swc[[2, 3, 4]] = swc[[2, 3, 4]] - [min_x, min_y, min_z]
        swc.to_csv(os.path.join(swc_in_DB, swc_file), sep=" ", header=None, index=None)

        # swc to tif
        # img_shape = (int(swc[4].max()) + 1, int(swc[3].max()) + 1, int(swc[2].max()) + 1)
        max_x, max_y, max_z = swc[[2, 3, 4]].max()
        x_end, y_end, z_end = int(max_x+1), int(max_y+1), int(max_z+1)
        img_shape = (z_end, y_end, x_end)


        branch_mask_file = os.path.join(temp_branch_mask_dir, f"{swc_file.split('.')[0]}.tif")
        if(not os.path.exists(branch_mask_file)):
            swc2img(os.path.join(swc_in_DB, swc_file), img_shape, branch_mask_file)
            branch_mask = tiff.imread(branch_mask_file)
            branch_mask = np.where(branch_mask > 0, 1, 0).astype("uint8")
            tiff.imwrite(branch_mask_file, branch_mask, compression=5)
        # mip = branch_mask.max(axis=0)*255
        # tiff.imwrite(os.path.join(temp_mip_dir, f"{swc_file.split('.')[0]}.tif"), mip)
        swc = origin_swc
        swc.to_csv(os.path.join(swc_in_DB, swc_file), sep=" ", header=None, index=None)

        current_soma_block_file = os.path.join(temp_soma_block_dir, f"image_{swc_file.split('.')[0]}_0000.tif")
        if(not os.path.exists(current_soma_block_file)):
            soma_markers = pd.read_csv(soma_markers_file, sep=',',
                                       comment='#',
                                       names=['n', 'orderinfo', 'name', 'comment', 'z', 'x', 'y', 'pixmax', 'intensity',
                                              'sdev', 'volsize', 'mass', 'comment1', 'comment2', 'comment3', 'color_r',
                                              'color_g', 'color_b'])
            # print(soma_markers, int(swc_file.split('.')[0])+423)
            swc_soma_pos = soma_markers[soma_markers['name'] == int(swc_file.split('.')[0])+375][['x', 'y', 'z']]
            if(swc_soma_pos.shape[0] == 0):
                print(f"{swc_file} has no soma marker")
                continue
            # print(swc_soma_pos)
            swc_soma_pos = swc_soma_pos.mean().values.astype("int")

            x_start, y_start, z_start = swc_soma_pos[0] - soma_block_size[0]//2, swc_soma_pos[1] - soma_block_size[1]//2, swc_soma_pos[2] - soma_block_size[2]//2
            x_end, y_end, z_end = x_start + soma_block_size[0], y_start + soma_block_size[1], z_start + soma_block_size[2]
            x_start, y_start, z_start = max(0, x_start), max(0, y_start), max(0, z_start)
            x_end, y_end, z_end = min(img.shape[2], x_end), min(img.shape[1], y_end), min(img.shape[0], z_end)
            soma_block = img[z_start:z_end, y_start:y_end, x_start:x_end]
            tiff.imwrite(current_soma_block_file, soma_block, compression=5)


    # seg soma
    soma_predictor.predict_from_files(temp_soma_block_dir, temp_soma_seg_dir,
                 save_probabilities=False, overwrite=False,
                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                 folder_with_segs_from_prev_stage=None)

    total_branch_mask = np.zeros_like(img).astype(np.float32)
    for swc_file in tqdm(swc_files):
        branch_mask_file = os.path.join(temp_branch_mask_dir, f"{swc_file.split('.')[0]}.tif")
        branch_mask = tiff.imread(branch_mask_file)
        branch_mask = np.where(branch_mask > 0, 1, 0).astype(np.float32)
        swc = pd.read_csv(os.path.join(swc_in_DB, swc_file), sep=" ", header=None, comment="#")

        x_start, y_start, z_start = swc[[2, 3, 4]].min()
        x_start, y_start, z_start = int(x_start-1), int(y_start-1), int(z_start-1)
        x_end, y_end, z_end = swc[[2, 3, 4]].max()
        x_end, y_end, z_end = int(x_end+1), int(y_end+1), int(z_end+1)

        # total_branch_mask[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1] += branch_mask
        total_branch_mask[int(z_start):int(z_end), int(y_start):int(y_end), int(x_start):int(x_end)] += branch_mask

        soma_seg_file = os.path.join(temp_soma_seg_dir, f"image_{swc_file.split('.')[0]}.tif")
        soma_seg = tiff.imread(soma_seg_file).astype(np.float32)
        current_soma_block_file = os.path.join(temp_soma_block_dir, f"{swc_file.split('.')[0]}.tif")
        soma_markers = pd.read_csv(soma_markers_file, sep=',',
                                   comment='#',
                                   names=['n', 'orderinfo', 'name', 'comment', 'z', 'x', 'y', 'pixmax', 'intensity',
                                          'sdev', 'volsize', 'mass', 'comment1', 'comment2', 'comment3', 'color_r',
                                          'color_g', 'color_b'])
        # print(soma_markers, int(swc_file.split('.')[0])+423)
        swc_soma_pos = soma_markers[soma_markers['name'] == int(swc_file.split('.')[0]) + 375][['x', 'y', 'z']]
        # print(swc_soma_pos)
        swc_soma_pos = swc_soma_pos.mean().values.astype("int")
        x_start, y_start, z_start = swc_soma_pos[0] - soma_block_size[0] // 2, swc_soma_pos[1] - soma_block_size[1] // 2, \
                                    swc_soma_pos[2] - soma_block_size[2] // 2
        x_end, y_end, z_end = x_start + soma_block_size[0], y_start + soma_block_size[1], z_start + soma_block_size[2]
        x_start, y_start, z_start = max(0, x_start), max(0, y_start), max(0, z_start)
        x_end, y_end, z_end = min(img.shape[2], x_end), min(img.shape[1], y_end), min(img.shape[0], z_end)
        total_branch_mask[z_start:z_end, y_start:y_end, x_start:x_end] -= soma_seg

    total_maks_dir = "/data2/kfchen/tracing_ws/branch_seg/mask_DB"
    os.makedirs(total_maks_dir, exist_ok=True)

    total_mask_file = os.path.join(total_maks_dir, "total_mask.tif")
    total_branch_mask = np.where(total_branch_mask > 0, 1, 0).astype("uint8")
    tiff.imwrite(total_mask_file, total_branch_mask, compression=5)
    total_branch_mask_mip_file = os.path.join(temp_mip_dir, "total_branch_mask_mip.tif")
    mask_mip = total_branch_mask.max(axis=0) * 255
    img_mip = img.max(axis=0)
    mip = np.concatenate([mask_mip, img_mip], axis=1)
    tiff.imwrite(total_branch_mask_mip_file, mip)









