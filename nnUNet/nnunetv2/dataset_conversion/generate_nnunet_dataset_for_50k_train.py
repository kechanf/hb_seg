import time
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import tifffile
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from tqdm import tqdm
import os
import pandas as pd
from tifffile import imread, imwrite
import numpy as np
from skimage.measure import block_reduce
from pylib.file_io import load_image
import cc3d
from nnUNet.scripts.adaptive_gamma import find_resolution, adaptive_augment_gamma, down_sample, de_fluorophore_gamma_augment
from simple_swc_tool.soma_detection import simple_get_soma
import concurrent.futures
import subprocess

def augment_gamma(data_sample, gamma_range=(0.5, 2), epsilon=1e-7, per_channel=False,
                  retain_stats=False, p=1):
    """Function directly copied from batchgenerators"""
    if(np.random.random() > p):
        return data_sample
    # gamma = np.random.uniform(gamma_range[0], 1)
    gamma = 0.5
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    return data_sample
    return None

def generate_single_nnunet_data(img_file, mask_file, id, nnunet_dataset_id, image_dir, mask_dir, image_tr, label_tr, raw_info_df):

    target_name = f'image_{nnunet_dataset_id:03d}'

    img = tifffile.imread(os.path.join(image_dir, img_file))
    mask = tifffile.imread(os.path.join(mask_dir, mask_file))

    img_size = img.shape
    spacing = find_resolution(raw_info_df, img_file)
    # spacing = (1, float(spacing) / 1000, float(spacing) / 1000)
    spacing = (1,1,1)

    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype('uint8')

    tifffile.imwrite(os.path.join(image_tr, target_name + '_0000.tif'), img)
    save_json({'spacing': spacing}, os.path.join(image_tr, target_name + '.json'))

    mask = np.where(mask > 0, 1, 0).astype("uint8")
    tifffile.imwrite(os.path.join(label_tr, target_name + '.tif'), mask, compression='zlib')
    save_json({'spacing': spacing}, os.path.join(label_tr, target_name + '.json'))

    return {
        'ID': id,
        'full_name': img_file,
        'nnunet_name': target_name,
        'spacing': spacing,
        'img_size': img_size,
        'raw_path': os.path.join(image_dir, img_file)
    }

def generate_train_data(image_dir, mask_dir, image_tr, label_tr,
                        raw_info_df, name_mapping_file):
    data = {
        'ID': [],
        'full_name': [],
        'nnunet_name': [],
        'spacing': [],
        'img_size': [],
        'raw_path': []
    }

    img_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    img_files.sort()
    mask_files.sort()
    ids = [int(im.split('_')[0].split('.')[0]) for im in img_files]
    nnunet_dataset_file_count = -1
    nnunet_dataset_ids = []

    progress_bar = tqdm(total=len(img_files), desc="Copying img", unit="file")

    for (img_file, mask_file, id) in zip(img_files, mask_files, ids):
        nnunet_dataset_file_count = nnunet_dataset_file_count + 1
        nnunet_dataset_ids.append(nnunet_dataset_file_count)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Preparing future tasks
        futures = [
            executor.submit(generate_single_nnunet_data, img_file, mask_file, id, nnunet_dataset_id, image_dir, mask_dir, image_tr,
                            label_tr, raw_info_df)
            for img_file, mask_file, id, nnunet_dataset_id in zip(img_files, mask_files, ids, nnunet_dataset_ids)
        ]

        # Collecting results as they become available
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if(result):
                data['ID'].append(result['ID'])
                data['full_name'].append(result['full_name'])
                data['nnunet_name'].append(result['nnunet_name'])
                data['spacing'].append(result['spacing'])
                data['img_size'].append(result['img_size'])
                data['raw_path'].append(result['raw_path'])

    df = pd.DataFrame(data)
    df = df.sort_values(by='ID')
    if(os.path.exists(name_mapping_file)):
        os.remove(name_mapping_file)
    df.to_csv(name_mapping_file, index=False)
    progress_bar.close()

    done_files = os.listdir(image_tr)
    done_files = [file for file in done_files if file.endswith('.tif')]

    # generate_dataset_json(
    #     join(nnUNet_raw, dataset_name),
    #     {0: 'mi'},
    #     {'background': 0, 'neuron': 1},
    #     len(done_files),
    #     '.tif'
    # )

def generate_dataset(dataset_name):
    nnUNet_raw = r"/data2/kfchen/nnUNet/nnUNet_raw"
    os.makedirs(nnUNet_raw, exist_ok=True)

    # images_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/raw"
    images_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/raw"
    seg_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/mask"

    raw_info_path = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"

    imagestr = os.path.join(nnUNet_raw, dataset_name, "imagesTr")
    labelstr = os.path.join(nnUNet_raw, dataset_name, "labelsTr")
    imagests = os.path.join(nnUNet_raw, dataset_name, "imagesTs")
    name_mapping_file = os.path.join(nnUNet_raw, dataset_name, "name_mapping.csv")

    if not os.path.exists(os.path.join(nnUNet_raw, dataset_name)):
        os.makedirs(os.path.join(nnUNet_raw, dataset_name))
        os.makedirs(imagestr)
        os.makedirs(labelstr)
        os.makedirs(imagests)

    raw_info_df = pd.read_csv(raw_info_path, header=None, encoding='gbk')
    generate_train_data(images_dir, seg_dir, imagestr, labelstr, raw_info_df, name_mapping_file)

def move_test_data(dataset_name):
    img_dir_tr = "/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/imagesTr"
    img_dir_ts = "/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/imagesTs"
    mask_dir_tr = "/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/labelsTr"

    new_img_tr = "/data/kfchen/nnUNet/nnUNet_raw/" + dataset_name + "/imagesTr"
    new_img_ts = "/data/kfchen/nnUNet/nnUNet_raw/" + dataset_name + "/imagesTs"
    new_mask_tr = "/data/kfchen/nnUNet/nnUNet_raw/" + dataset_name + "/labelsTr"

    # move test
    val_files = [f for f in os.listdir(img_dir_ts) if f.endswith('.tif')]

    source_tif_dir = new_img_tr
    target_tif_dir = new_img_ts

    for val_file in val_files:
        source_tif = os.path.join(source_tif_dir, val_file)
        target_tif = os.path.join(target_tif_dir, val_file)
        source_json_file = os.path.join(source_tif_dir, val_file.replace("_0000.tif", ".json"))
        target_json_file = os.path.join(target_tif_dir, val_file.replace("_0000.tif", ".json"))
        # shutil.copy(source_tif, target_tif)
        # shutil.copy(source_json_file, target_json_file)
        # 剪切
        shutil.move(source_tif, target_tif)
        shutil.move(source_json_file, target_json_file)


    # kill non shared files
    dir_pairs = [(img_dir_tr, new_img_tr), (img_dir_ts, new_img_ts), (mask_dir_tr, new_mask_tr)]
    for source_dir, target_dir in dir_pairs:
        source_files = [f for f in os.listdir(source_dir)]
        target_files = [f for f in os.listdir(target_dir)]
        for target_file in target_files:
            if target_file not in source_files:
                os.remove(os.path.join(target_dir, target_file))

    json_file = "/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/dataset.json"
    new_json_file = "/data/kfchen/nnUNet/nnUNet_raw/" + dataset_name + "/dataset.json"
    if(os.path.exists(new_json_file)):
        os.remove(new_json_file)
    shutil.copy(json_file, new_json_file)

if __name__ == '__main__':
    dataset_name = 'Dataset191_full_res'
    generate_dataset(dataset_name)
    move_test_data(dataset_name)

    dataset_id = dataset_name[7:10]
    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id} -c 3d_fullres --verify_dataset_integrity"
    print(cmd)
    def process_path(pstr):
        return pstr.replace('(', '\(').replace(')', '\)')
    subprocess.run(process_path(cmd), stdout=subprocess.DEVNULL, shell=True)








