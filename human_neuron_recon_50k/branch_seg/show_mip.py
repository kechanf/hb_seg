import os
import tifffile as tiff
import numpy as np
import cv2

img_dir = "/data2/kfchen/nnUNet/nnUNet_raw/Dataset202_hb_branch/imagesTr"
label_dir  = "/data2/kfchen/nnUNet/nnUNet_raw/Dataset202_hb_branch/labelsTr"
mip_dir = "/data2/kfchen/nnUNet/nnUNet_raw/Dataset202_hb_branch/mip"
for img_file in os.listdir(img_dir):
    if(not img_file.endswith(".tif")):
        continue
    img_path = os.path.join(img_dir, img_file)
    label_file = img_file.replace("_0000", "")
    label_path = os.path.join(label_dir, label_file)
    mip_file = img_file.replace("image", "mip")
    mip_path = os.path.join(mip_dir, mip_file)

    if(os.path.exists(mip_path)):
        continue
    if(not os.path.exists(label_path)):
        continue

    img = tiff.imread(img_path)
    label = tiff.imread(label_path)
    if(np.max(label) == 0):
        continue

    img_mip = np.max(img, axis=0)
    lab_mip = np.max(label, axis=0) * 255

    # 转换数据类型为 uint8
    img_mip = img_mip.astype(np.uint8)
    lab_mip = lab_mip.astype(np.uint8)

    # 如果原图是灰度图，将其转换为 3 通道 BGR 图像
    if len(img_mip.shape) == 2:
        img_rgb = cv2.cvtColor(img_mip, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = img_mip

    # 创建一个全黑的图像，作为标签的彩色蒙版（这里使用红色表示标签区域）
    mask_color = np.zeros_like(img_rgb)
    # 在红色通道填充标签区域（OpenCV中图像通道顺序为BGR）
    mask_color[..., 2] = lab_mip

    # 设置透明度
    alpha = 0.5
    # 叠加蒙版：原图 * 1.0 + 标签蒙版 * alpha
    overlay = cv2.addWeighted(img_rgb, 1.0, mask_color, alpha, 0)

    # 保存叠加后的图像
    tiff.imwrite(mip_path, overlay)

