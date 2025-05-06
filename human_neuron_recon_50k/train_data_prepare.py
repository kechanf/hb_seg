import os
import shutil
from tqdm import tqdm
from pylib.file_io import load_image
import numpy as np
import tifffile
from skimage.transform import resize
import pandas as pd
from joblib import Parallel, delayed
import subprocess
from simple_swc_tool.swc_radius_estimator import SWC_Radius_Estimator

v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"

def rescale_swc(swc_file, output_swc_file, xy_resolution, z_resolution, target_voxel_resolution):
    with open(swc_file, "r") as f:
        lines = f.readlines()
    with open(output_swc_file, "w") as f:
        for line in lines:
            if line.startswith("#"):
                f.write(line)
            else:
                parts = line.strip().split()
                x = float(parts[2]) * xy_resolution / target_voxel_resolution
                y = float(parts[3]) * xy_resolution / target_voxel_resolution
                z = float(parts[4]) * z_resolution / target_voxel_resolution
                r = float(parts[5]) * xy_resolution / target_voxel_resolution
                f.write(f"{parts[0]} {parts[1]} {x} {y} {z} {r} {parts[6]}\n")

def resample_swc(swc_file, output_swc_file=None, resample_step=4):
    if((swc_file == output_swc_file) or (output_swc_file is None)):
        output_swc_file_tmp = swc_file.replace(".swc", "_tmp.swc")
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" "{v3d_path}" -x resample_swc -f resample_swc -i {swc_file} -o {output_swc_file_tmp} -p {str(resample_step)}'
        cmd = cmd.replace('\\', '/')
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
        os.remove(swc_file)
        os.rename(output_swc_file_tmp, swc_file)
    else:
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" "{v3d_path}" -x resample_swc -f resample_swc -i {swc_file} -o {output_swc_file} -p {str(resample_step)}'
        cmd = cmd.replace('\\', '/')
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)

source_double_checked_manual_recon_dir = "/data/kfchen/trace_ws/paper_trace_result/manual/double_checked_anno_swc"
v3d_img_root = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"
v3d_img_ext = ".v3draw"

v3d_img_files = []
for root, dirs, files in os.walk(v3d_img_root):
    if(not "human_brain_data_v3draw" in root):
        continue
    for file in files:
        if file.endswith(v3d_img_ext):
            v3d_img_files.append(os.path.join(root, file))

double_checked_manual_recon_files = [os.path.join(source_double_checked_manual_recon_dir, f)
                                     for f in os.listdir(source_double_checked_manual_recon_dir) if f.endswith(".swc")]

v3d_img_id_map = {}
for img_file in v3d_img_files:
    img_id = int(os.path.basename(img_file).split("_")[0])
    v3d_img_id_map[img_id] = img_file

doubel_checked_manual_recon_id_map = {}
for recon_file in double_checked_manual_recon_files:
    img_id = int(os.path.basename(recon_file).split("_")[0])
    doubel_checked_manual_recon_id_map[img_id] = recon_file

# shared ids
shared_ids = set(v3d_img_id_map.keys()) & set(doubel_checked_manual_recon_id_map.keys())
shared_ids = list(shared_ids)
random_pick_num = 2
shared_ids = np.random.choice(shared_ids, random_pick_num, replace=False)

recon_meta_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
recon_meta = pd.read_excel(recon_meta_file)
# print(recon_meta["id"])
recon_meta["id"] = recon_meta["cell_id"]
recon_meta["id"] = recon_meta["id"].astype(int)
recon_meta = recon_meta[recon_meta["id"].isin(shared_ids)]
# print(f"recon_meta: {recon_meta.shape}")

target_voxel_resolution = 250 # nm
traget_root = "/data2/kfchen/tracing_ws/human_neuron_recon_50k"
# target_img_dir = os.path.join(traget_root, "images_" + str(target_voxel_resolution) + "nm")
target_img_dir = "/data2/kfchen/human_neuron_seg_50k/nnUNet_raw/Dataset101_hb/imagesTr"
target_swc_dir = os.path.join(traget_root, "swc_" + str(target_voxel_resolution) + "nm")
os.makedirs(target_img_dir, exist_ok=True)
os.makedirs(target_swc_dir, exist_ok=True)

def prepare_data(img_file, swc_file, target_img_file, target_swc_file, xy_resolution, z_resolution, target_voxel_resolution):
    if(os.path.exists(target_img_file) and os.path.exists(target_swc_file)):
        return
    img = load_image(img_file)[0]
    img = img.astype(np.float32)
    origin_shape = img.shape
    xy_resolution, z_resolution = float(xy_resolution), float(z_resolution)
    rescaled_250um_shape = (int(origin_shape[0] * z_resolution / 250),
                            int(origin_shape[1] * xy_resolution / 250),
                            int(origin_shape[2] * xy_resolution / 250))
    img = resize(img, rescaled_250um_shape, order=1, mode="reflect", anti_aliasing=True)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype("uint8")
    tifffile.imwrite(target_img_file, img)

    rescale_swc(swc_file, target_swc_file, xy_resolution, z_resolution, target_voxel_resolution)
    resample_swc(target_swc_file, resample_step=10)
    swc_radius_estimator = SWC_Radius_Estimator(target_swc_file, target_img_file)

Parallel(n_jobs=10)(delayed(prepare_data)(v3d_img_id_map[img_id], doubel_checked_manual_recon_id_map[img_id],
                                            os.path.join(target_img_dir, "image_{:05d}_0000.tif".format(img_id)),
                                            os.path.join(target_swc_dir, "{:05d}.swc".format(img_id)),
                                            recon_meta[recon_meta["id"] == img_id]["xy_resolution"].values[0],
                                            recon_meta[recon_meta["id"] == img_id]["z_resolution"].values[0],
                                            target_voxel_resolution) for img_id in tqdm(shared_ids))









