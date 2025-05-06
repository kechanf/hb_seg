import numpy as np
from skimage import io
import os

def split_3d_image(image, patch_size=(256, 256, 256), overlap=0.1):
    """
    将三维图像切分为指定大小的图像块，图像块之间有重叠，大小不足的自动补足
    :param image: 输入的三维图像 (D, H, W)
    :param patch_size: 图像块大小 (depth, height, width)
    :param overlap: 图像块之间的重叠比例 (0.0 ~ 1.0)
    :return: 切分后的图像块列表
    """
    # 图像块大小
    depth, height, width = patch_size
    # 图像块之间的重叠大小
    overlap_depth = int(depth * overlap)
    overlap_height = int(height * overlap)
    overlap_width = int(width * overlap)

    # 补足图像大小
    D, H, W = image.shape
    D_pad = (D // depth + 1) * depth - D
    H_pad = (H // height + 1) * height - H
    W_pad = (W // width + 1) * width - W
    image = np.pad(image, ((0, D_pad), (0, H_pad), (0, W_pad)), mode="constant")

    # 切分图像
    patches = []
    for d in range(0, D, depth - overlap_depth):
        for h in range(0, H, height - overlap_height):
            for w in range(0, W, width - overlap_width):
                patch = image[d:d + depth, h:h + height, w:w + width]
                patches.append(patch)

    return patches


# ----------------------
# 使用示例
# ----------------------
cropped_mask_root = "/data2/kfchen/tracing_ws/branch_seg/mask"
cropped_img_root = "/data2/kfchen/tracing_ws/branch_seg/cropped_dataset/img"
cropped_mask_root = "/data2/kfchen/tracing_ws/branch_seg/cropped_dataset/mask"

img_file = "/data2/kfchen/tracing_ws/soma_seg/tif_image/P00122-T001-R001-S006-B1_0000.tif"
mask_file = "/data2/kfchen/tracing_ws/branch_seg/mask_DB/total_mask_122.tif"
img_name = os.path.basename(img_file).split(".")[0]
cropped_mask_root_dir = os.path.join(cropped_mask_root, img_name)
cropped_img_root_dir = os.path.join(cropped_img_root, img_name)
os.makedirs(cropped_mask_root_dir, exist_ok=True)
os.makedirs(cropped_img_root_dir, exist_ok=True)


img = io.imread(img_file)
mask = io.imread(mask_file)
img_patches = split_3d_image(img, patch_size=(256, 256, 256), overlap=0.1)
mask_patches = split_3d_image(mask, patch_size=(256, 256, 256), overlap=0.1)
mask_patches = [np.where(mask_patche, 1, 0).astype('uint8') for mask_patche in mask_patches]
#
# # 输出结果
# print(f"切分后的图像块数量: {len(img_patches)}")
# # save
# for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
#     img_patch_name = f"{img_name}_{i:04d}.tif"
#     mask_patch_name = f"{img_name}_{i:04d}.tif"
#     io.imsave(os.path.join(cropped_mask_root_dir, img_patch_name), img_patch)
#     io.imsave(os.path.join(cropped_img_root_dir, mask_patch_name), mask_patch, compression=5)



target_img_dir = "/data2/kfchen/nnUNet/nnUNet_raw/Dataset202_hb_branch/imagesTr"
target_mask_dir = "/data2/kfchen/nnUNet/nnUNet_raw/Dataset202_hb_branch/labelsTr"
os.makedirs(target_img_dir, exist_ok=True)
os.makedirs(target_mask_dir, exist_ok=True)
file_number = 0
for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
    file_number += 1
    img_patch_name = f"image_{file_number:04d}_0000.tif"
    mask_patch_name = f"image_{file_number:04d}.tif"
    io.imsave(os.path.join(target_img_dir, img_patch_name), img_patch)
    io.imsave(os.path.join(target_mask_dir, mask_patch_name), mask_patch, compression=5)

# 手动处理一下json
#
'''
nnUNetv2_plan_and_preprocess -d 202 -c 3d_fullres --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 202 3d_fullres 0 -num_gpus 1

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i /data2/kfchen/nnUNet/nnUNet_raw/Dataset202_hb_branch/imagesTr -o /data2/kfchen/tracing_ws/branch_seg/seg/122 -d 202 -c 3d_fullres -f 0
'''

