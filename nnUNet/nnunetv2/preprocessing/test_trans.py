from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
import time
import os

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

def normalize_mip(img):
    # Normalize image
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

def generate_mip_images(seg, seg_resized):
    # Generate MIPs
    seg_mip = np.max(seg, axis=0)
    seg_resized_mip = np.max(seg_resized, axis=0)

    # Normalize images
    seg_mip = normalize_mip(seg_mip)
    seg_resized_mip = normalize_mip(seg_resized_mip)
    # print(seg_mip.shape, seg_resized_mip.shape)

    # Add space between images
    gap = 10  # width of the gap in pixels
    height = max(seg_mip.shape[0], seg_resized_mip.shape[0])
    gap_array = np.zeros((height, gap), dtype=np.uint8)
    # print(seg_mip.shape, gap_array.shape, seg_resized_mip.shape)

    # Concatenate images with the gap
    if seg_mip.shape[0] < height:
        padding_height = height - seg_mip.shape[0]
        padding = np.zeros((padding_height, seg_mip.shape[1]), dtype=np.uint8)
        seg_mip = np.concatenate([seg_mip, padding], axis=0)
    # print(seg_mip.shape, gap_array.shape, seg_resized_mip.shape)

    if seg_resized_mip.shape[0] < height:
        padding_height = height - seg_resized_mip.shape[0]
        padding = np.zeros((padding_height, seg_resized_mip.shape[1]), dtype=np.uint8)
        seg_resized_mip = np.concatenate([seg_resized_mip, padding], axis=0)

    # print(seg_mip.shape, gap_array.shape, seg_resized_mip.shape)
    mip_compare = np.concatenate([seg_mip, gap_array, seg_resized_mip], axis=1)

    # Save the combined image
    rand_file_path = f"/data/kfchen/nnUNet/temp_mip/train/rand_{time.time()}.png"
    plt.imsave(rand_file_path, mip_compare, cmap='gray')

class TestTransform(AbstractTransform):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, **data_dict):
        print(f"shape {data_dict['seg'].shape}, {data_dict['data'].shape}")
        # shape(2, 1, 48, 263, 263), (2, 1, 48, 263, 263)

        num_batches = data_dict['seg'].shape[0]
        scaled_data = []
        scaled_seg = []

        for batch in range(num_batches):
            data = data_dict['data'][batch]
            seg = data_dict['seg'][batch]

            # dir = "/data/kfchen/nnUNet/temp_mip/train"
            # rand_file_name = f"rand_{time.time()}.png"
            # rand_file_path = os.path.join(dir, rand_file_name)
            # data_mip = np.max(data, axis=0)
            # data_mip = ((data_mip - data_mip.min()) / (data_mip.max() - data_mip.min()) * 255).astype(np.uint8)
            # plt.imsave(rand_file_path, data_mip, cmap='gray')


            # print(f"data shape {data.shape}, {self.patch_size}")
            # data shape (48, 263, 263), [48, 224, 224]
            # data_resized, seg_resized = resize_to_patch(data, seg, self.patch_size)
            # print(f"data_resized shape {data_resized.shape}, data shape {data.shape}")

            seg = np.where(seg > 0.5, 1, 0)

            # print(f"seg_resized shape {seg_resized.shape}, seg shape {seg.shape}")
            # seg_resized shape (48, 224, 224), seg shape (48, 263, 263)

            generate_mip_images(data, seg)

            scaled_data.append(data)
            scaled_seg.append(seg)


        # 更新字典中的数据
        data_dict['data'] = np.array(scaled_data)
        data_dict['seg'] = np.array(scaled_seg)

        return data_dict