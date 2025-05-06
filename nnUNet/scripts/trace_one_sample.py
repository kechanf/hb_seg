from v3dpy.loaders import Raw, PBD
import numpy as np
import tifffile
from human_neuron_recon_50k.recon_from_segment import process_pipeline
import pandas as pd


v3d_img_file = "/data2/kfchen/tracing_ws/temp/to_xiaoxuan/example.v3dpbd"
input_img_file = "/data2/kfchen/tracing_ws/temp/to_xiaoxuan/input/image_777_0000"
# pbd = PBD()
# img = pbd.load(v3d_img_file)[0]
# img = img.astype(np.float32)
# img = np.flip(img, axis=0)
# img = (img - img.min()) / (img.max() - img.min()) * 255
# img = img.astype("uint8")
# tifffile.imsave(input_img_file + ".tif", img)
'''
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i /data2/kfchen/tracing_ws/temp/to_xiaoxuan/input -o /data2/kfchen/tracing_ws/temp/to_xiaoxuan/output -d 191 -c 3d_fullres -f 0
'''
work_dir = "/data2/kfchen/tracing_ws/temp/to_xiaoxuan"
file_name = "image_777.tif"
meta_info = {
    "xy_resolution": 500,
    "z_resolution": 1000,
    "soma_x": 450,
    "soma_y": 800, # 和v3d中soma位置一致，和tif在neurontube中的位置相反
    "soma_z": 47,
}
# to df
meta_info = pd.DataFrame([meta_info])
process_pipeline(work_dir, file_name, meta_info)
