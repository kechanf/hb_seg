from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
import time
import os

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import tifffile
import cc3d

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


def pad_to_patch_size(image, patch_size):
    """
    将三维图像填充到指定的 patch_size。

    参数:
    image (numpy.ndarray): 输入的三维图像。
    patch_size (tuple of int): 需要填充到的尺寸 (depth, height, width)。

    返回:
    numpy.ndarray: 填充后的图像。
    """
    # 计算当前图像每个维度需要填充的大小
    padding = []
    for i, size in enumerate(patch_size):
        if image.shape[i] < size:
            # 计算前后需要填充的数量
            total_pad = size - image.shape[i]
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding.append((pad_before, pad_after))
        else:
            padding.append((0, 0))  # 如果图像的维度已经满足或超过，不需要填充

    # 使用np.pad进行填充，填充模式选择'constant'，填充值默认为0
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    return padded_image

class SimpleTransform(AbstractTransform):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, **data_dict):
        # print(f"shape {data_dict['seg'].shape}, {data_dict['data'].shape} in TestTransform")
        # shape(2, 1, 48, 263, 263), (2, 1, 48, 263, 263)

        num_batches = data_dict['seg'].shape[0]
        scaled_data = []
        scaled_seg = []

        for batch in range(num_batches):
            data = data_dict['data'][batch]
            seg = data_dict['seg'][batch]

            # print(f"data shape {data.shape}, {self.patch_size}")
            # data shape (48, 263, 263), [48, 224, 224]
            # data_resized, seg_resized = resize_to_patch(data, seg, self.patch_size)
            # print(f"data_resized shape {data_resized.shape}, data shape {data.shape}")

            # print(f"data.shape, seg.shape: {data.shape, seg.shape} in TestTransform0000000000000")

            seg = np.where(seg > 0.5, 1, 0)
            seg = pad_to_patch_size(seg[0], self.patch_size)
            seg = np.expand_dims(seg, axis=0)

            # print(f"seg_resized shape {seg_resized.shape}, seg shape {seg.shape}")
            # seg_resized shape (48, 224, 224), seg shape (48, 263, 263)

            # generate_mip_images(data, seg)

            # print(f"data.shape, seg.shape: {data.shape, seg.shape} in TestTransform")
            # _, cc_num = cc3d.connected_components(seg[0], connectivity=26, return_N=True)
            # if(cc_num > 1):
            #     print("fuckkkkkkkkkkkkkkkkkkkkk in SimpleTransform")
            #     # generate_mip_images(data, seg)

            scaled_data.append(data)
            scaled_seg.append(seg)


        # 更新字典中的数据
        data_dict['data'] = np.array(scaled_data)
        data_dict['seg'] = np.array(scaled_seg)

        return data_dict

class TestTransform(AbstractTransform):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, **data_dict):
        # print(f"shape {data_dict['seg'].shape}, {data_dict['data'].shape} in TestTransform")
        # shape(2, 1, 48, 263, 263), (2, 1, 48, 263, 263)

        num_batches = data_dict['target'].shape[0]
        scaled_data = []
        scaled_seg = []

        for batch in range(num_batches):
            data = data_dict['data'][batch]
            seg = data_dict['target'][batch]

            # _, cc_num = cc3d.connected_components(seg[0], connectivity=26, return_N=True)
            # if(cc_num > 1):
            #     print("fuckkkkkkkkkkkkkkkkkkkkk in TestTransform")
            #     # generate_mip_images(data, seg)

            scaled_data.append(data)
            scaled_seg.append(seg)


        # 更新字典中的数据
        data_dict['data'] = np.array(scaled_data)
        data_dict['seg'] = np.array(scaled_seg)

        return data_dict

if __name__ == "__main__":
    data_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset165_human_brain_resized_10k_ptls/imagesTr"
    seg_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset165_human_brain_resized_10k_ptls/labelsTr"

    data_files = os.listdir(data_dir)
    seg_files = os.listdir(seg_dir)

    data_files = [file for file in data_files if file.endswith(".tif")]
    seg_files = [file for file in seg_files if file.endswith(".tif")]

    data_files.sort()
    seg_files.sort()

    for i in range(len(data_files)):
        data_file = data_files[i]
        seg_file = seg_files[i]

        data_path = os.path.join(data_dir, data_file)
        seg_path = os.path.join(seg_dir, seg_file)

        data = np.array(tifffile.imread(data_path))
        seg = np.array(tifffile.imread(seg_path))

        if(not data.shape == (64, 256, 256) or not seg.shape == (64, 256, 256)):
            print(data_path, seg_path)
            print(data.shape, seg.shape)
            continue

