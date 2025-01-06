import os
import tifffile
import numpy as np
import shutil
from pylib.file_io import load_image
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm

source_img_root = "/PBshare/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/v3dpbd_cells"
target_data_root = "/PBshare/SEU-ALLEN/Users/KaifengChen/Human_Seg_Pro/sample_data_50k"
target_v3d_img_dir = os.path.join(target_data_root, "img_v3dpbd")
target_tif_img_dir = os.path.join(target_data_root, "img_tif")
target_mip_dir = os.path.join(target_data_root, "mip")

source_img_list = []
# walk
for root, dirs, files in os.walk(source_img_root):
    for file in files:
        if file.endswith(".v3dpbd"):
            source_img_list.append(os.path.join(root, file))

def current_task(source_img_file, target_v3d_img_file, target_tif_img_file, target_mip_file):
    img_name = os.path.basename(source_img_file).split('.')[0]
    if(os.path.exists(target_v3d_img_file)):
        return

    shutil.copy(source_img_file, target_v3d_img_file)
    img = load_image(source_img_file)[0]
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype('uint8')
    tifffile.imwrite(target_tif_img_file, img)

    mip = img.max(axis=0)
    tifffile.imwrite(target_mip_file, mip)

print("Total images: ", len(source_img_list))
sample_step = 250
source_img_list = [f for f in source_img_list if int(os.path.basename(f).split('.')[0]) % sample_step == 0]
print("Sample images: ", len(source_img_list))
joblib.Parallel(n_jobs=8)(
    joblib.delayed(current_task)(
        source_img_file,
        os.path.join(target_v3d_img_dir, os.path.basename(source_img_file)),
        os.path.join(target_tif_img_dir, os.path.basename(source_img_file).replace('.v3dpbd', '.tif')),
        os.path.join(target_mip_dir, os.path.basename(source_img_file).replace('.v3dpbd', '.png'))
    ) for source_img_file in tqdm(source_img_list)
)

# for source_img_file in tqdm(source_img_list):
#     current_task(
#         source_img_file,
#         os.path.join(target_v3d_img_dir, os.path.basename(source_img_file)),
#         os.path.join(target_tif_img_dir, os.path.basename(source_img_file).replace('.v3dpbd', '.tif')),
#         os.path.join(target_mip_dir, os.path.basename(source_img_file).replace('.v3dpbd', '.png'))
#     )

