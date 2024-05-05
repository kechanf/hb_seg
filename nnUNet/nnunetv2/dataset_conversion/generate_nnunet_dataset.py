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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def delete_non_shared_files(img_folder, mask_folder):
    print(f"Deleting non-shared files in {img_folder} and {mask_folder}...")
    img_files = set(os.listdir(img_folder))
    mask_files = set(os.listdir(mask_folder))

    non_shared_img_files = img_files - mask_files
    non_shared_mask_files = mask_files - img_files

    # 删除非共有的图片文件
    for file_name in non_shared_img_files:
        file_path = os.path.join(img_folder, file_name)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

    # 删除非共有的掩膜文件
    for file_name in non_shared_mask_files:
        file_path = os.path.join(mask_folder, file_name)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

def get_spacing(img_path, raw_info_path):
    import pandas as pd
    if os.path.isabs(img_path):
        img_path = os.path.basename(img_path)
    if("_P" in img_path and "_T" in img_path and "_R" in img_path):
        # find _P
        preffix = img_path.split('_')
        spacing = 1000
        for i in range(len(preffix)):
            # print(preffix[i])
            if(len(preffix[i]) <= 2):
                continue
            if(preffix[i][0] == 'R' and preffix[i][1:].isdigit()):
                spacing = int(preffix[i][1:])
                break
        return (1, float(spacing/1000), float(spacing/1000))
    else:
        try:
            raw_info = pd.read_excel(raw_info_path)
            preffix = img_path.split('/')[-1].split('.')[0]
            spacing = raw_info[raw_info['number'] == float(preffix)][['resolution']].values[0]
            return (1, float(spacing[0]/1000), float(spacing[0]/1000))
        except:
            return (1, 1, 1)

def copy_mask_files(src_path, dest_path):
    mask = imread(src_path)
    mask[mask > 0] = 1
    imwrite(dest_path, mask.astype("uint8"))

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

def get_sorted_files(directory, suffix='.v3draw'):
    v3draw_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix) and "_i" not in file and "_p" not in file:
                v3draw_files.append(os.path.join(root, file))

    v3draw_files.sort()
    return v3draw_files

def find_muti_soma_marker_file(file_name, muti_soma_marker_folder):
    if("tif" in file_name):
        file_name = file_name[:-4]
    if("v3draw" in file_name):
        file_name = file_name[:-7]
    if(os.path.exists(os.path.join(muti_soma_marker_folder, os.path.splitext(file_name)[0] + '.marker'))):
        return os.path.join(muti_soma_marker_folder, os.path.splitext(file_name)[0] + '.marker')
    ID = file_name.split('_')[0]
    # find the soma marker according to the ID
    file_names = os.listdir(muti_soma_marker_folder)
    for file in file_names:
        if(int(file.split('_')[0]) == int(ID)):
            return os.path.join(muti_soma_marker_folder, file)
    return None

def del_none_tif_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('.tif'):
                os.remove(os.path.join(root, file))

def generate_train_data(images_dir, seg_dir, imagestr, labelstr,
                        mutisoma_marker_path, label_info, csv_path,
                        generate_muti_soma=0, debug=False):
    """

    Args:
        images_dir:
        seg_dir:
        imagestr:
        labelstr:
        mutisoma_marker_path: 0 for skip the image with muti soma marker, 1 for skip single soma cases, 2 for no skip.
                              Default is 0
        label_info:
        csv_path:
        generate_muti_soma:
        debug:

    Returns:

    """
    data = {
        'ID': [],
        'full_name': [],
        'nnunet_name': [],
        'spacing': [],
        'img_size': [],
        'v3draw_path': []
    }

    delete_non_shared_files(images_dir, seg_dir)


    images = subfiles(images_dir, suffix='.tif', sort=True, join=False)
    segs = subfiles(seg_dir, suffix='.tif', sort=True, join=False)
    # endswith('.tif')
    images = [im for im in images if im.endswith('.tif')]
    segs = [se for se in segs if se.endswith('.tif')]
    # sort
    images.sort()
    segs.sort()
    ids = [int(im[:-4].split('/')[-1].split('_')[0]) for im in images]
    # print(len(images), len(segs), len(ids))
    print(images)
    print(segs)
    print(ids)

    if(debug):
        images = images[:10]
        segs = segs[:10]
        ids = ids[:10]
    # print(images)

    progress_bar = tqdm(total=len(images), desc="Copying img", unit="file")

    for (im, se, id) in zip(images, segs, ids):
        file_name = im.split('/')[-1]
        muti_soma_marker_path = find_muti_soma_marker_file(file_name, mutisoma_marker_path)
        if ((generate_muti_soma==0) and (not (muti_soma_marker_path is None))): # skip the image with muti soma marker
            print(f"Skip {im} because of {muti_soma_marker_path}")
            continue
        elif(generate_muti_soma==1 and (muti_soma_marker_path)): # skip single soma cases
            print(f"Skip {im} because of {muti_soma_marker_path}")
            continue


        target_name = f'image_{id:03d}'
        # print(target_name)

        # spacing = get_spacing(im, raw_info_path)
        resol = label_info[label_info['number'] == int(im.split('/')[-1].split('.')[0])][['resolution']].values[0]
        spacing = (1, resol[0] / 1000, resol[0] / 1000)

        img = imread(join(images_dir, im))
        img_size = img.shape
        img = augment_gamma(img, gamma_range=(0.5, 0.5))
        img = img.astype(np.uint8)
        tifffile.imwrite(join(imagestr, target_name + '_0000.tif'), img)
        save_json({'spacing': spacing}, join(imagestr, target_name + '.json'))

        img = imread(join(seg_dir, se))
        img = np.where(img > 0, 1, 0).astype("uint8")
        tifffile.imwrite(join(labelstr, target_name + '.tif'), img)
        save_json({'spacing': spacing}, join(labelstr, target_name + '.json'))

        temp_im = im.split('/')[-1]
        if (im.endswith('.tif')):
            temp_im = im[:-4]
        elif (im.endswith('.v3draw')):
            temp_im = temp_im[:-7]

        data['full_name'].append(temp_im)
        data['nnunet_name'].append(target_name)
        data["spacing"].append(spacing)
        data["img_size"].append(img_size)
        data["ID"].append(int(temp_im.split('_')[0]))
        data["v3draw_path"].append(im)

        progress_bar.update(1)

    df = pd.DataFrame(data)
    df = df.sort_values(by='ID')
    df.to_csv(csv_path, index=False)
    progress_bar.close()

    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        {0: 'mi'},
        {'background': 0, 'neuron': 1},
        len(images),
        '.tif'
    )

def generate_test_data(test_source, imagests, raw_info_path, mutisoma_marker_path, csv_path, generate_muti_soma=0, debug=True):
    data = {
        'ID': [],
        'full_name': [],
        'nnunet_name': [],
        'spacing': [],
        'img_size': [],
        'v3draw_path': []
    }

    images = get_sorted_files(test_source, suffix='.v3draw')
    ids = [int(im.split('/')[-1].split('_')[0]) for im in images]
    gt_files = get_sorted_files("/data/kfchen/nnUNet/gt_swc", suffix='.swc')
    gt_ids = [int(im.split('/')[-1].split('_')[0]) for im in gt_files]
    shared_ids = list(set(ids).intersection(set(gt_ids)))
    images = [im for im in images if int(im.split('/')[-1].split('_')[0]) in shared_ids]

    if(debug):
        images = images[:10]
        ids = ids[:10]

    progress_bar = tqdm(total=len(images), desc="Copying img", unit="file")
    for im, id in zip(images, ids):
        progress_bar.update(1)
        target_name = f'image_{(id):03d}'

        file_name = im.split('/')[-1]
        muti_soma_marker_path = find_muti_soma_marker_file(file_name, mutisoma_marker_path)
        if ((generate_muti_soma == 0) and (
        not (muti_soma_marker_path is None))):  # skip the image with muti soma marker
            print(f"Skip {im} because of {muti_soma_marker_path}")
            continue
        elif (generate_muti_soma == 1 and (muti_soma_marker_path)):  # skip single soma cases
            print(f"Skip {im} because of {muti_soma_marker_path}")
            continue

        img_size = [1, 1, 1]
        spacing = get_spacing(im, raw_info_path)

        try:
            img = load_image(im, False)[0].astype("uint8")
            img_size = img.shape
            if (not os.path.exists(join(imagests, target_name + '_0000.tif'))):
                # print(img.shape)
                img = augment_gamma(img)
                # block reduce
                factor = 2
                img = block_reduce(img, block_size=(factor, factor, factor), func=np.max)
                imwrite(join(imagests, target_name + '_0000.tif'), img.astype("uint8"))

                # spacing file!
                save_json({'spacing': spacing}, join(imagests, target_name + '.json'))
        except Exception as e:
            if (os.path.exists(join(imagests, target_name + '_0000.tif'))):
                os.remove(join(imagests, target_name + '_0000.tif'))
            if (os.path.exists(join(imagests, target_name + '.json'))):
                os.remove(join(imagests, target_name + '.json'))
            # with open(f"error_log.txt", "a") as file:
            #     file.write(f"An error occurred: {e} at {im}\n")

        temp_im = im.split('/')[-1]
        if (im.endswith('.v3draw')):
            temp_im = temp_im[:-7]
        data['full_name'].append(temp_im)
        data['nnunet_name'].append(target_name)
        data["spacing"].append(spacing)
        data["img_size"].append(img_size)
        data["ID"].append(int(id))
        data["v3draw_path"].append(join(test_source, im))

    df = pd.DataFrame(data)
    df = df.sort_values(by='ID')
    df.to_csv(csv_path, index=False)

    progress_bar.close()

def copy_gt_files(source_path, dest_path, mutisoma_marker_path, generate_muti_soma=0, debug=False):
    if (os.path.exists(dest_path)):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)
    file_list = os.listdir(source_path)
    if(debug):
        file_list = file_list[:10]
    for file_name in file_list:
        if (file_name.endswith('.eswc')):
            muti_soma_marker_path = find_muti_soma_marker_file(file_name, mutisoma_marker_path)
            if ((generate_muti_soma == 0) and
                    (not (muti_soma_marker_path is None))):  # skip the image with muti soma marker
                print(f"Skip {file_name} because of {muti_soma_marker_path}")
                continue
            if(os.path.exists(os.path.join(dest_path, file_name)) or
                    os.path.exists(os.path.join(dest_path, file_name[:-5] + '.swc'))):
                continue
            shutil.copyfile(os.path.join(source_path, file_name), os.path.join(dest_path, file_name))
            with open(os.path.join(dest_path, file_name), 'r') as file:
                new_lines = []
                lines = file.readlines()
                split_lines = [line.split() for line in lines]
                for i in range(len(split_lines)):
                    if(split_lines[i][0] == '#'):
                        continue
                    split_lines[i] = split_lines[i][0:7]
                    new_line = str(' '.join(split_lines[i])) + '\n'
                    # print(new_line)
                    new_lines.append(new_line)
                with open(os.path.join(dest_path, file_name), 'w') as file:
                    file.writelines(new_lines)
            # rename
            if (file_name.endswith('.eswc')):
                os.rename(os.path.join(dest_path, file_name), os.path.join(dest_path, file_name[:-5] + '.swc'))


def cp_gt_seg_file(file_name, gt_seg_dir, target_dir, max_workers=12):
    if (file_name.endswith('.tif')):
        img = tifffile.imread(os.path.join(gt_seg_dir, file_name))
        factor = 2
        img = block_reduce(img, block_size=(factor, factor, factor), func=np.max)
        img = img.astype("uint8")
        tifffile.imwrite(os.path.join(target_dir, file_name), img.astype("uint8"))




def cp_gt_seg(gt_seg_dir="/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/mask",
              target_dir="/data/kfchen/trace_ws/gt_seg_downsample/tif"):
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    file_names = [f for f in os.listdir(gt_seg_dir) if f.endswith('.tif')]
    partial_func = partial(cp_gt_seg_file, gt_seg_dir=gt_seg_dir, target_dir=target_dir)
    with Pool(12) as p:
        for _ in tqdm(p.imap(partial_func, file_names),
                      total=len(file_names), desc="to_v3dswc_folder", unit="file"):
            pass

if __name__ == '__main__':
    nnUNet_raw = r"/data/kfchen/nnUNet/nnUNet_raw"
    raw_info_path = "/home/kfchen/dataset/img/raw_info.xlsx"
    mutisoma_marker_path = r"/data/kfchen/nnUNet/nnUNet_raw/muti_soma_markers"
    label_info_path = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/label/label_info.xlsx"

    # dataset_name = 'Dataset101_human_brain_10000_ssoma_test'
    dataset_name = 'Dataset163_human_brain_resized_10k_source'
    # images_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/image"
    # seg_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/label"
    images_dir = "/data/kfchen/trace_ws/resized_dataset/img"
    seg_dir = "/data/kfchen/trace_ws/resized_dataset/lab"
    test_source = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw"
    imagestr = join(nnUNet_raw, dataset_name, "imagesTr")
    labelstr = join(nnUNet_raw, dataset_name, "labelsTr")
    imagests = join(nnUNet_raw, dataset_name, "imagesTs")
    csv_path = join(nnUNet_raw, dataset_name, "name_mapping.csv")
    gt_path = r"/PBshare/SEU-ALLEN/Projects/Human_Neurons/Version_human_dendrite_manual/V20230131/human_dendrite_852_eswc_V20230131/other492"

    if not os.path.exists(join(nnUNet_raw, dataset_name)):
        os.makedirs(join(nnUNet_raw, dataset_name))
        os.makedirs(imagestr)
        os.makedirs(labelstr)
        os.makedirs(imagests)

    label_info = pd.read_excel(label_info_path)
    generate_train_data(images_dir, seg_dir, imagestr, labelstr, mutisoma_marker_path, label_info, csv_path, generate_muti_soma=0, debug=False)
    # generate_test_data(test_source, imagests, raw_info_path, mutisoma_marker_path, csv_path, debug=False)
    # copy_gt_files(gt_path, r"/data/kfchen/nnUNet/gt_swc", mutisoma_marker_path, generate_muti_soma=0, debug=False)

    # nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
    # CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i /data/kfchen/nnUNet/nnUNet_raw/Dataset101_human_brain_10000_ssoma_test/imagesTs -o /data/kfchen/nnUNet/nnUNet_raw/result12k -d 158 -c 3d_fullres -f 0




