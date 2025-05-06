import tifffile as tiff
import os
from skimage.morphology import skeletonize_3d
from simple_swc_tool.big_neuron_tracers import neuTube_trace_file
from joblib import Parallel, delayed
from tqdm import tqdm

ws_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k"
seg_dir = os.path.join(ws_dir, "down_sampled_seg_with_soma")
skel_dir = os.path.join(ws_dir, "down_sampled_uint8")
swc_dir = os.path.join(ws_dir, "down_sampled_swcs")
os.makedirs(skel_dir, exist_ok=True)
os.makedirs(swc_dir, exist_ok=True)

def try_trace(seg_file, skel_file, swc_file):
    if not seg_file.endswith(".tif"):
        return
    if( not os.path.exists(seg_file)):
        print(f"File {seg_file} does not exist.")
        return
    if(os.path.exists(skel_file) and os.path.exists(swc_file)):
        print(f"File {seg_file} already processed.")
        return
    try:
        seg = tiff.imread(seg_file)
        # skel = skeletonize_3d(seg)
        skel = seg
        skel = (skel * 255).astype("uint8")
        tiff.imwrite(skel_file, skel)

        neuTube_trace_file(skel_file, swc_file)
    except Exception as e:
        print(f"Error processing {seg_file}: {e}")

seg_files = [
    f for f in os.listdir(seg_dir) if f.endswith(".tif")
]
Parallel(n_jobs=4)(
    delayed(try_trace)(
        os.path.join(seg_dir, seg_file),
        os.path.join(skel_dir, seg_file),
        os.path.join(swc_dir, seg_file.replace(".tif", ".swc"))
    ) for seg_file in tqdm(seg_files)
)

# for seg_file in os.listdir(seg_dir):
#     if seg_file.endswith(".tif"):
#         seg_path = os.path.join(seg_dir, seg_file)
#         skel_path = os.path.join(skel_dir, seg_file)
#         swc_path = os.path.join(swc_dir, seg_file.replace(".tif", ".swc"))
#
#         seg = tiff.imread(seg_path)
#         # skel = skeletonize_3d(seg)
#         skel = seg
#         skel = (skel * 255).astype("uint8")
#         tiff.imwrite(skel_path, skel)
#
#         neuTube_trace_file(skel_path, swc_



