import os
import sys
import subprocess
import numpy as np
import tifffile
import cc3d

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