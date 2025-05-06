import os
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import shutil
# This script is used to move the images from the source directory to the target directory


def task_1():
    meta_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
    meta = pd.read_csv(meta_file, encoding="gbk")
    ids = meta["cell_id"].tolist()
    print(len(ids))

    source_img_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/tif"
    target_img_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/imgs_to_release/img_8398"

    img_files = [f for f in os.listdir(source_img_dir) if f.endswith(".tif")]

    def current_task(img_file):
        img_id = int(img_file.split("_")[0])
        if (not img_id in ids):
            return
        # img_id = np.zfill(img_id, 5)
        img_id = str(img_id)
        source_img_path = os.path.join(source_img_dir, img_file)
        target_img_path = os.path.join(target_img_dir, img_id + ".tif")

        if (os.path.exists(target_img_path)):
            return

        # copy
        # os.system(f"cp {source_img_path} {target_img_path}")
        shutil.copy(source_img_path, target_img_path)

    # Parallel processing
    Parallel(n_jobs=8, backend="threading")(
        delayed(current_task)(img_file) for img_file in tqdm(img_files))


def task_2():
    