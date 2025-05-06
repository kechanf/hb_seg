import os
import pandas as pd
from pyparsing import ParseResults
from tifffile import tiffcomment
from pylib.file_io import load_image
import tifffile
import numpy as np
from nnUNet.nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
from joblib import Parallel, delayed
from human_neuron_recon_50k.recon_from_segment import process_pipeline
from tqdm import tqdm
import shutil
from nnUNet.scripts.mip import get_mip, get_mip_swc
from skimage.transform import resize
import matplotlib.pyplot as plt
import time
import random
from v3dpy.loaders import Raw, PBD
import threading

NUM_PROCESSERS = 4
NUM_GPUS = 1

def export_v3dpbd_to_tif(v3dpbd_file, target_dir):
    new_name = "image_" + os.path.basename(v3dpbd_file).replace(".v3dpbd", "_0000.tif")
    if(os.path.exists(os.path.join(target_dir, new_name))):
        return
    # print(f"Processing {v3dpbd_file}")
    # img = load_image(v3dpbd_file)[0]
    pbd = PBD()
    img = pbd.load(v3dpbd_file)[0]
    # print(img.shape)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype("uint8")

    tifffile.imwrite(os.path.join(target_dir, new_name), img)
    spacing_file = "/data2/kfchen/nnUNet/nnUNet_raw/Dataset191_full_res/imagesTr/image_000.json"
    shutil.copy(spacing_file, os.path.join(target_dir, new_name.replace("_0000.tif", ".json")))

def plot_recon_mip(img_file, seg_file, skel_file, swc_file, xy_resolution, mip_file):
    if(not os.path.exists(img_file) or not os.path.exists(seg_file) or not os.path.exists(swc_file) or not os.path.exists(skel_file)):
        return
    if(os.path.exists(mip_file)):
        return
    img = tifffile.imread(img_file)
    seg = tifffile.imread(seg_file)
    seg = np.where(seg > 0, 255, 0).astype(np.uint8)
    skel = tifffile.imread(skel_file)
    skel = np.where(skel > 0, 255, 0).astype(np.uint8)
    origin_img_shape = img.shape
    # rescaled_img = resize(img, (int(origin_img_shape[0]), int(origin_img_shape[1] * xy_resolution / 1000.0), int(origin_img_shape[2] * xy_resolution / 1000.0)), anti_aliasing=True, preserve_range=True, mode='constant', order=1)
    rescaled_img_bkg = np.zeros((int(origin_img_shape[0]), int(origin_img_shape[1] * xy_resolution / 1000.0), int(origin_img_shape[2] * xy_resolution / 1000.0)))

    img_mip = get_mip(img)
    seg_mip = get_mip(seg)
    skel_mip = get_mip(skel)
    skel_mip = np.flip(skel_mip, axis=0)
    # swc_mip = get_mip_swc(swc_file, rescaled_img)
    swc_mip_ignore_background = get_mip_swc(swc_file, rescaled_img_bkg, ignore_background=True)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(img_mip, cmap='gray')
    ax[0, 1].imshow(seg_mip, cmap='gray')
    ax[1, 0].imshow(skel_mip, cmap='gray')
    ax[1, 1].imshow(swc_mip_ignore_background)

    for a in ax.flatten():
        a.axis('off')

    plt.tight_layout()
    plt.savefig(mip_file)
    plt.close()

def process_predict(predictor_list, part_id, current_img_dir, current_seg_dir):
    predictor_list[part_id].predict_from_files(current_img_dir, current_seg_dir,
                 save_probabilities=False, overwrite=False,
                 num_processes_preprocessing=int(NUM_PROCESSERS), num_processes_segmentation_export=int(NUM_PROCESSERS),
                 folder_with_segs_from_prev_stage=None, num_parts=NUM_GPUS, part_id=part_id)

if __name__ == "__main__":
    img_root = "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/Cell_Images"
    recon_result = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/50k_recon"
    mip_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/50k_mip"
    temp_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp"
    seg_model_dir = "data2/kfchen/nnUNet/nnUNet_results/Dataset191_full_res/nnUNetTrainer__nnUNetPlans__3d_fullres"
    meta_info_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/meta_info_0205.xlsx"

    full_meta_info = pd.read_excel(meta_info_file)

    predictor_list = []
    for i in range(NUM_GPUS):
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', i),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        # /data2/kfchen/nnUNet/nnUNet_results/Dataset191_full_res/nnUNetTrainer__nnUNetPlans__3d_fullres
        predictor.initialize_from_trained_model_folder(
            "/data2/kfchen/nnUNet/nnUNet_results/Dataset191_full_res/nnUNetTrainer__nnUNetPlans__3d_fullres",
            use_folds=(0,),
            checkpoint_name='checkpoint_final.pth',
        )
        predictor_list.append(predictor)
    # predictor = nnUNetPredictor(
    #     tile_step_size=0.5,
    #     use_gaussian=True,
    #     use_mirroring=True,
    #     perform_everything_on_device=True,
    #     device=torch.device('cuda', 0),
    #     verbose=False,
    #     verbose_preprocessing=False,
    #     allow_tqdm=True
    # )
    # # /data2/kfchen/nnUNet/nnUNet_results/Dataset191_full_res/nnUNetTrainer__nnUNetPlans__3d_fullres
    # predictor.initialize_from_trained_model_folder(
    #     "/data2/kfchen/nnUNet/nnUNet_results/Dataset191_full_res/nnUNetTrainer__nnUNetPlans__3d_fullres",
    #     use_folds=(0,),
    #     checkpoint_name='checkpoint_final.pth',
    # )

    img_files = []
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if file.endswith(".v3dpbd"):
                img_files.append(os.path.join(root, file))
    random.shuffle(img_files)

    files_to_be_processed = img_files
    batch_size = 100
    os.makedirs(recon_result, exist_ok=True)
    os.makedirs(mip_dir, exist_ok=True)
    start_time = time.time()
    while True:
        print(f"-------------------{len(os.listdir(recon_result))} / {len(img_files)} done-------------------")
        time_cost = time.time() - start_time
        print(f"Time cost (h): {time_cost/3600.0}, Time left (h):{(len(img_files) - len(os.listdir(recon_result)))/len(os.listdir(recon_result)) * time_cost / 3600.0}")

        if(len(files_to_be_processed) == 0):
            break

        current_files_to_be_processed = []
        current_img_dir = os.path.join(temp_dir, "image")
        current_seg_dir = os.path.join(temp_dir, "0_seg")
        os.makedirs(current_img_dir, exist_ok=True)
        os.makedirs(current_seg_dir, exist_ok=True)

        if(len(os.listdir(current_seg_dir)) < batch_size):
            for img_file in files_to_be_processed:
                if(len(current_files_to_be_processed) >= batch_size):
                    break
                if(os.path.join(recon_result, os.path.basename(img_file).replace(".v3dpbd", ".swc")) not in os.listdir(recon_result)):
                    file_name = os.path.basename(img_file).replace(".v3dpbd", ".swc")
                    meta_info = full_meta_info[full_meta_info["cell_id"] == int(file_name.split('.')[0])]
                    if(meta_info['soma_x'].values[0] == '-'):
                        continue
                    if((meta_info['patient_number'].values[0] == 'P00089' and meta_info["tissue_block_number"].values[0] == 'T002')):
                        current_files_to_be_processed.append(img_file)
                    elif(meta_info['patient_number'].values[0] == 'P00089' and meta_info["fresh_perfusion"].values[0] == 1):
                        current_files_to_be_processed.append(img_file)

            for img_file in current_files_to_be_processed:
                files_to_be_processed.remove(img_file)


            # v3d to tif
            Parallel(n_jobs=NUM_PROCESSERS*4)(delayed(export_v3dpbd_to_tif)(img_file, current_img_dir) for img_file in tqdm(current_files_to_be_processed, desc="exporting v3dpbd to tif"))

            # segmentation
            # for part_id in range(NUM_GPUS):
            #     process_predict(predictor_list, part_id, current_img_dir, current_seg_dir)
            # 创建线程列表
            threads = []

            # 启动多线程
            for part_id in range(NUM_GPUS):
                thread = threading.Thread(
                    target=process_predict,
                    args=(predictor_list, part_id, current_img_dir, current_seg_dir)
                )
                thread.start()
                threads.append(thread)

            # 等待所有线程完成
            for thread in threads:
                thread.join()

        # tracing
        for seg_file in os.listdir(current_seg_dir):
            new_name = seg_file.replace("_0000", "").replace("image_", '')
            os.rename(os.path.join(current_seg_dir, seg_file), os.path.join(current_seg_dir, new_name))


        work_dir = temp_dir
        # print(f"Processing {work_dir}")
        # work_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/"
        # origin_seg_dir = os.path.join(work_dir, "origin_seg")# "origin_seg"

        seg_dir = os.path.join(work_dir, "0_seg")

        file_names = [f for f in os.listdir(seg_dir) if f.endswith('.tif')]
        if(os.path.exists(os.path.join(work_dir, "8_estimated_radius_swc"))):
            for swc_file in os.listdir(os.path.join(work_dir, "8_estimated_radius_swc")):
                if(swc_file.endswith(".swc")):
                    file_names.remove(swc_file.replace(".swc", ".tif"))
        Parallel(n_jobs=NUM_PROCESSERS)(delayed(process_pipeline)(
            work_dir, file_name, full_meta_info[full_meta_info["cell_id"] == int(file_name.split('.')[0])]
        ) for file_name in tqdm(file_names, desc="tracing"))


        # for file_name in tqdm(file_names, desc="tracing"):
        #     process_pipeline(work_dir, file_name, full_meta_info[full_meta_info["cell_id"] == int(file_name.split('.')[0])])

        # final_swc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/8_estimated_radius_swc"
        final_swc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/9_soma_g_cut_swc"
        skel_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/3_skel_with_soma"
        for swc_file in os.listdir(final_swc_dir):
            # shutil.move(os.path.join(final_swc_dir, swc_file), os.path.join(recon_result, swc_file))
            shutil.copy(os.path.join(final_swc_dir, swc_file), os.path.join(recon_result, swc_file))
            # print(f"Copying {swc_file} to {recon_result}")

        Parallel(n_jobs=NUM_PROCESSERS)(delayed(plot_recon_mip)
            (os.path.join(current_img_dir, "image_" + swc_file.replace(".swc", "_0000.tif")),
             os.path.join(current_seg_dir, swc_file.replace(".swc", ".tif")),
             os.path.join(skel_dir, swc_file.replace(".swc", ".tif")),
             os.path.join(recon_result, swc_file),
             full_meta_info[full_meta_info["cell_id"] == int(swc_file.split(".")[0])]["xy_resolution"].values[0],
             os.path.join(mip_dir, swc_file.replace(".swc", ".png"))
        )
        for swc_file in tqdm(os.listdir(recon_result), desc="plotting mip"))

        break
        # 删除文件夹
        shutil.rmtree(temp_dir)



