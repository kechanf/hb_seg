import pandas as pd
from pylib.file_io import load_image
import numpy as np
import os
from skimage.draw import line_aa
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import tifffile

test_source = r"/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw"

dir_root = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/10847_auto_v1.4_12697_add"
# dir_root = "/data/kfchen/10847"
list_10847 = dir_root + r"/list_10847.xlsx"
list_traced = dir_root + r"/list_traced.xlsx"
swc_root = dir_root + r"/swc"
mip_root = dir_root + r"/mip"
mip_swc_root = dir_root + r"/mip_swc"

def maybe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
if(os.path.exists(mip_root) == False):
    maybe_mkdir(mip_root)
if(os.path.exists(mip_swc_root) == False):
    maybe_mkdir(mip_swc_root)

class swcPoint:
    def __init__(self, sample_number, structure_identifier,
                 x_position, y_position, z_position, radius, parent_sample):
        self.n = sample_number
        self.si = 0#structure_identifier
        self.x = x_position
        self.y = y_position
        self.z = z_position
        self.r = radius
        self.p = parent_sample
        self.s = [] # sons
        self.fn = -1 # fiber number
        self.conn = [] # connect points in other fiber
        self.mp = [] # match point in other swc
        self.neighbor = [] # neighbor closer than a distance. store neighbor number and connect info. as [d, bool]
        # self.isend = False
        self.ishead = False
        self.istail = False
        self.swcNeig = [] # neighbor closer than a distance.
        self.swcMatchP = []
        self.i = 0
        self.visited = 0
        self.pruned = False
        self.depth = 0




    def EndCheck(self):
        return self.ishead or self.istail


    def Printswc(self):
        print("n=%d, si=%d, x=%f, y=%f, z=%f, r=%f, p=%d, s=%s, fn=%d, neighbor=%s, mp=%s"
              %(self.n, self.si, self.x, self.y, self.z, self.r, self.p, str(self.s),
                self.fn, str(self.neighbor), str(self.mp)))

    def Writeswc(self, filepath, swcPoint_list,
                 reversal=False, limit=[256, 256, 128],
                 overlay=False, number_offset=0):
        if(reversal):
            line = "%d %d %f %f %f %f %d\n" %(
                self.n + number_offset, self.si, self.x,
                limit[1] - self.y,
                self.z, self.r, self.p + number_offset
            )
        else:
            line = "%d %d %f %f %f %f %d\n" %(
                self.n + number_offset, self.si, self.x,
                self.y,
                self.z, self.r, self.p + number_offset
            )
        if (overlay and os.path.exists(filepath)):
            # print("!!!!!!")
            os.remove(filepath)
        file_handle = open(filepath, mode="a")
        file_handle.writelines(line)
        file_handle.close()

class swcP_list:
    def __init__(self):
        self.p = []
        self.count = 0
def Readswc_v2(swc_name):
    point_l = swcP_list()
    with open(swc_name, 'r' ) as f:
        lines = f.readlines()

    swcPoint_number = -1
    # swcPoint_list = []
    point_list = []
    list_map = np.zeros(500000)

    for line in lines:
        if(line[0] == '#'):
            continue

        temp_line = line.split()
        # print(temp_line)
        point_list.append(temp_line)

        swcPoint_number = swcPoint_number + 1
        list_map[int(temp_line[0])] = swcPoint_number

    # print(point_list)
    swcPoint_number = 0
    for point in point_list:
        swcPoint_number = swcPoint_number + 1
        point[0] = swcPoint_number # int(point[0])
        point[1] = int(point[1])
        point[2] = float(point[2])
        point[3] = float(point[3])
        point[4] = float(point[4])
        point[5] = float(point[5])
        point[6] = int(point[6])
        if(point[6] == -1):
            pass
        else:
            point[6] = int(list_map[int(point[6])]) + 1

    # swcPoint_list.append(swcPoint(0,0,0,0,0,0,0)) # an empty point numbered 0
    point_l.p.append(swcPoint(0,0,0,0,0,0,0))

    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        point_l.p.append(temp_swcPoint)
    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        if not temp_swcPoint.p == -1:
            # parent = swcPoint_list[int(temp_swcPoint.p)]
            parent = point_l.p[int(temp_swcPoint.p)]
            parent.s.append(temp_swcPoint.n)
        if(point[0] == 1):
            point_l.p[int(point[0])].depth = 0
        else:
            point_l.p[int(point[0])].depth = parent.depth + 1
        # point_l.p.append(temp_swcPoint)
    # for i in range(1, 10):
    #     print(point_l.p[i].s)

    return point_l # (swcPoint_list)

def get_sorted_files(directory, suffix='.v3draw'):
    v3draw_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix) and "_i" not in file and "_p" not in file:
                v3draw_files.append(os.path.join(root, file))

    v3draw_files.sort()
    return v3draw_files

def get_mip(image, projection_direction='xy'):
    if projection_direction == 'xy':
        projection_axes = 0
    elif projection_direction == 'xz':
        projection_axes = 1
    elif projection_direction == 'yz':
        projection_axes = 2
    else:
        raise ValueError("Invalid projection direction. Choose from 'xy', 'xz', or 'yz'.")

    mip = np.max(image, axis=projection_axes).astype("uint8")
    return mip

def get_mip_swc(swc_file, image, projection_direction='xy'):
    if projection_direction == 'xy':
        projection_axes = 0
    elif projection_direction == 'xz':
        projection_axes = 1
    elif projection_direction == 'yz':
        projection_axes = 2
    else:
        raise ValueError("Invalid projection direction. Choose from 'xy', 'xz', or 'yz'.")
    background = get_mip(image, projection_direction).astype("uint8")
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    point_l = Readswc_v2(swc_file)

    color = (0, 0, 255)
    thickness = 1

    if (projection_axes == 0):
        cv2.circle(background, (int(point_l.p[1].x), int(point_l.p[1].y)), 3, color, -1)
    elif (projection_axes == 1):
        cv2.circle(background, (int(point_l.p[1].x), int(point_l.p[1].z)), 3, color, -1)
    elif (projection_axes == 2):
        cv2.circle(background, (int(point_l.p[1].y), int(point_l.p[1].z)), 3, color, -1)

    for p in point_l.p:
        if (p.n == 0 or p.n == 1): continue
        if (p.p == 0 or p.p == -1): continue
        x, y, z = p.x, p.y, p.z
        px, py, pz = point_l.p[p.p].x, point_l.p[p.p].y, point_l.p[p.p].z
        # y, py = background.shape[1] - y, background.shape[1] - py

        x, y, z = int(x), int(y), int(z)
        px, py, pz = int(px), int(py), int(pz)


        if (projection_axes == 0):
            # draw a line between two points
            cv2.line(background, (x, y), (px, py), color, thickness)
        elif (projection_axes == 1):
            cv2.line(background, (x, z), (px, pz), color, thickness)
        elif (projection_axes == 2):
            cv2.line(background, (y, z), (py, pz), color, thickness)

    return background

def find_swc(v3d_file, swc_root):
    swc_file = v3d_file.split('/')[-1].replace('.v3draw', '.swc')
    id = swc_file.split('_')[0]
    # walk through all the files in the directory to find the swc file
    for root, dirs, file_names in os.walk(swc_root):
        for file_name in file_names:
            if(id in file_name):
                full_path = os.path.join(root, file_name)
                # print(full_path)
                return full_path
    return None



def concat_image_file(filename, folder_path, concat_folder, dir_list):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        if os.path.exists(os.path.join(concat_folder, filename)):
            return

        img_list = {}
        for i, path in enumerate(folder_path):
            img_path = os.path.join(path, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_list[i] = img
            else:
                img_list[i] = np.zeros((10, 10, 3), dtype=np.uint8)

        fig, axs = plt.subplots(1, len(img_list), figsize=(15, 5))
        for i, img in img_list.items():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img)
            axs[i].set_title(f"{filename} - {dir_list[i]}")
            axs[i].axis('off')

        plt.subplots_adjust(wspace=0.1)
        plt.savefig(os.path.join(concat_folder, filename), dpi=800)
        plt.close()

def concat_images(max_workers=12):
    folder_path_prefix = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/10847_auto_"
    # dir_list = ["v1.0", "v1.1", "v1.2", "v1.3"]
    dir_list = ["v1.4_e1000+250_noptls", "v1.4_e1000+250_ptls"]
    folder_path = [folder_path_prefix + dir + "/mip_swc" for dir in dir_list]

    # 新文件夹路径
    concat_folder = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/concat_mip_ptls_e1000"
    os.makedirs(concat_folder, exist_ok=True)

    file_list = []
    for path in folder_path:
        file_list.append(os.listdir(path))
    # 去除重复
    file_list = list(set(file_list[0]).intersection(*file_list))

    process_bar = tqdm(total=len(file_list), desc='Processing')

    # 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务到线程池
        futures = [executor.submit(concat_image_file, filename, folder_path, concat_folder, dir_list) for filename in
                   file_list]
        # 等待所有任务完成
        for future in as_completed(futures):
            process_bar.update(1)

    process_bar.close()

def main_mip(only_10847=False):
    images = get_sorted_files(test_source, suffix='.v3draw')

    if(only_10847):
        df_list = pd.read_excel(list_10847)['Cell ID'].values
        df_traced = pd.read_excel(list_traced)['ID'].values

    num = 0
    process_bar = tqdm(total=len(images), desc='Processing')

    for im in images:
        process_bar.update(1)
        # print(image)
        if ("_i" in im or "_p" in im):
            continue
        ID = int(im.split('/')[-1].split('_')[0])

        mip_save_path = os.path.join(mip_root, str(ID) + '.png')
        mip_swc_save_path = os.path.join(mip_swc_root, str(ID) + '.png')
        swc_file = find_swc(im, swc_root)
        if (swc_file is None):
            continue

        if(only_10847):
            if (ID not in df_list or ID not in df_traced):
                print(ID)
                if (os.path.exists(swc_file)):
                    os.remove(swc_file)
                continue

        if (os.path.exists(mip_save_path) and os.path.exists(mip_swc_save_path)):
            continue

        try:
            img = load_image(im, False)[0].astype("uint8")
            # normalize to 0-255
            img = (img - img.min()) / (img.max() - img.min()) * 255

            # save mip
            # mip = get_mip(img)
            mip_swc = get_mip_swc(swc_file, img)

            # save
            # cv2.imwrite(mip_save_path, mip)
            cv2.imwrite(mip_swc_save_path, mip_swc)
            num = num + 1

            # if(num>10):
            #     break
        except:
            print("Error: ", im)
            continue

    process_bar.close()


def calculate_mip(image_path):
    """计算给定图像文件的最大强度投影。"""
    image = tifffile.imread(image_path)
    return np.max(image, axis=0)


def compare_seg_mip_image_file(filename, folder1, folder2, output_folder):
    """处理单个图像文件并保存对比图。"""
    img_path1 = os.path.join(folder1, filename)
    img_path2 = os.path.join(folder2, filename)

    # 计算每个文件的MIP
    mip1 = calculate_mip(img_path1)
    mip2 = calculate_mip(img_path2)

    # 创建对比图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(filename)  # 设置大标题为文件名

    axs[0].imshow(mip1, cmap='gray')
    axs[0].set_title("origin")
    axs[0].axis('off')

    axs[1].imshow(mip2, cmap='gray')
    axs[1].set_title("ptls")
    axs[1].axis('off')

    plt.subplots_adjust(wspace=0.1)

    # 保存对比图
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path)
    plt.close()


def compare_seg_mip_images(folder1, folder2, output_folder, max_workers=12):
    if(os.path.exists(output_folder) == False):
        os.makedirs(output_folder)
    """比较两个文件夹中相同文件名的3D TIF图像的MIP，并保存对比图。"""
    # 获取两个文件夹中的图像文件名
    files1 = {file for file in os.listdir(folder1) if file.endswith('.tif')}
    files2 = {file for file in os.listdir(folder2) if file.endswith('.tif')}

    # 找出两个文件夹中都存在的文件
    common_files = list(files1.intersection(files2))

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 使用线程池处理每个图像文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用tqdm创建进度条
        results = list(tqdm(
            executor.map(compare_seg_mip_image_file, common_files, [folder1] * len(common_files), [folder2] * len(common_files),
                         [output_folder] * len(common_files)), total=len(common_files), desc="Processing Images"))


if __name__ == '__main__':
    main_mip()
    # concat_images()
    # compare_seg_mip_images("/data/kfchen/trace_ws/result500_e1000+250_noptls/tif",
    #                        "/data/kfchen/trace_ws/result500_e1000+250_ptls/tif",
    #                        "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/compare_seg_mip_<v1.4_result500_e1000+250_noptls>_vs_<v1.4_/result500_e1000+250_ptls>")



