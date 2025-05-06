import os
import numpy as np
import tifffile as tiff
from joblib import Parallel, delayed
from tqdm import tqdm

img_dir = "/data2/kfchen/nnUNet/nnUNet_raw/Dataset202_hb_branch/imagesTr"
label_dir = "/data2/kfchen/nnUNet/nnUNet_raw/Dataset202_hb_branch/labelsTr"
seg_dir = "/data2/kfchen/nnUNet/nnUNet_results/Dataset202_hb_branch/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
mip_dir = "/data2/kfchen/nnUNet/nnUNet_results/Dataset202_hb_branch/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/mip"
os.makedirs(mip_dir, exist_ok=True)

img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
def current_task(img_file, label_file, seg_file, mip_file):
    # if(os.path.exists(mip_file)):
    #     return
    if(not os.path.exists(seg_file) or not os.path.exists(label_file)):
        return

    img = tiff.imread(img_file)
    label = tiff.imread(label_file)
    seg = tiff.imread(seg_file)

    mips = [
        np.max(img, axis=0),
        np.max(label, axis=0) * 255,
        np.max(seg, axis=0) * 255
    ]
    mip = np.concatenate(mips, axis=1)
    tiff.imwrite(mip_file, mip)

Parallel(n_jobs=8)(delayed(current_task)(
    os.path.join(img_dir, f),
    os.path.join(label_dir, f.replace("_0000", "")),
    os.path.join(seg_dir, f.replace("_0000", "")),
    os.path.join(mip_dir, f.replace("_0000", ""))
)for f in tqdm(img_files))
'''

'''

