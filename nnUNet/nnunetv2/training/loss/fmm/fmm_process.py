import cc3d
import numpy as np
import os
# from simple_swc_tool.soma_detection import simple_get_soma
from nnunetv2.training.loss.fmm.fmm_path import compute_fast_marching
import tifffile
import subprocess
import sys
import matplotlib.pyplot as plt

def compute_centroid(mask):
    # 使用 cc3d 对3D mask进行连通区域标记
    labels = cc3d.connected_components(mask)  # 默认情况下, cc3d 会返回一个与 mask 同形状的标记数组

    # 初始化最大连通块的体积为0和其标签
    max_volume = 0
    max_label = 0

    # 找出最大的连通块
    for label in np.unique(labels):
        if label == 0:  # 跳过背景
            continue
        volume = (labels == label).sum()
        if volume > max_volume:
            max_volume = volume
            max_label = label

    # 如果没有找到连通块，则返回None
    if max_label == 0:
        return None

    # 计算最大连通块的重心
    coords = np.argwhere(labels == max_label)
    centroid = coords.mean(axis=0)

    return tuple(centroid)

def simple_get_soma(img, img_path, temp_path=r"/home/kfchen/temp_tif", v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    file_name = os.path.basename(img_path)
    # print(file_name)
    in_tmp = os.path.join(temp_path, file_name + 'temp.tif')
    tifffile.imwrite(in_tmp, (img * 255).astype(np.uint8))
    out_tmp = in_tmp.replace('.tif', '_gsdt.tif')

    if (sys.platform == "linux"):
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x gsdt -f gsdt -i {in_tmp} -o {out_tmp} -p 0 1 0 1.5'
        cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
        # print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
    else:
        cmd = f'{v3d_path} /x gsdt /f gsdt /i {in_tmp} /o {out_tmp} /p 0 1 0 1.5'
        # print(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if(not os.path.exists(out_tmp)):
        return None
    gsdt = tifffile.imread(out_tmp).astype(np.uint8)
    gsdt = np.flip(gsdt, axis=1)

    max_gsdt = np.max(gsdt)
    gsdt[gsdt <= max_gsdt * 0.95] = 0
    gsdt[gsdt > max_gsdt * 0.95] = 1

    centroid = compute_centroid(gsdt)

    if (os.path.exists(out_tmp)): os.remove(out_tmp)
    if (os.path.exists(in_tmp)): os.remove(in_tmp)
    del out_tmp, in_tmp

    return centroid


def muti_cc_image_check(image):
    # 计算连通块
    labels_out, N = cc3d.connected_components(image, connectivity=26, return_N=True)

    # 检查连通块数量
    if N > 1:
        # print(f"图像中有 {N} 个连通块。")
        # # 计算每个连通块的大小
        # component_sizes = np.bincount(labels_out.flatten())[1:]  # 忽略背景
        # for i, size in enumerate(component_sizes, 1):
        #     print(f"连通块 {i} 的大小为: {size}")

        # if(N > 5):
        #     # 保存MIP图
        #     save_mip_image(image)
        #     return True
        return False
    else:
        return False


def save_mip_image(image, filename='', temp_dir="/data/kfchen/nnUNet/temp_mip"):
    if(filename == ''):
        filename = str(np.random.randint(0, 100000)) + '.png'
    mip = np.max(image, axis=0)
    plt.imshow(mip, cmap='gray')
    plt.axis('off')
    plt.title('Maximum Intensity Projection')
    plt.savefig(os.path.join(temp_dir, filename))
    plt.close()
    print(f"MIP图已保存为 {filename}")

def get_fmm_from_img(img, temp_path=r"/home/kfchen/temp_tif", source=None):
    if(muti_cc_image_check(img)):
        return None, None
    # rand path
    random_path = np.random.randint(0, 100000) + (img.shape[0] * img.shape[1] * img.shape[2])
    img = img.astype(np.uint8)
    if(source is None):
        source = simple_get_soma(img, str(random_path), temp_path=temp_path)
    # print(f"source: {source}")
    if(source is None):
        return None, None
    source = tuple(int(i) for i in source)
    distance_image, predecessor_image = compute_fast_marching(img, source, img.shape)
    return predecessor_image, source