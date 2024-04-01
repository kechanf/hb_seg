import pandas as pd
from pylib.file_io import load_image
import numpy as np
import os
from skimage.draw import line_aa
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

test_source = r"/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw"

dir_root = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/10847_auto_v1.3_mutisoma"
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
    # walk through all the files in the directory to find the swc file
    for root, dirs, files in os.walk(swc_root):
        if swc_file in files:
            full_path = os.path.join(root, swc_file)
            # print(full_path)
            return full_path
    return None

def concat_images():
    folder_path_prefix = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/10847_auto_"
    dir_list = ["v1.0", "v1.1", "v1.2", "v1.3"]
    folder_path = [folder_path_prefix + dir + "/mip_swc" for dir in dir_list]

    # 新文件夹路径
    concat_folder = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/concat_mip"
    os.makedirs(concat_folder, exist_ok=True)

    process_bar = tqdm(total=len(os.listdir(folder_path[0])), desc='Processing')

    # 遍历第一个文件夹获取文件名
    for filename in os.listdir(folder_path[0]):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # 检查文件扩展名
            process_bar.update(1)
            if(os.path.exists(os.path.join(concat_folder, filename))):
                continue

            img_list = []

            for i in range(len(folder_path)):
                img_path = os.path.join(folder_path[i], filename)
                if(os.path.exists(img_path)):
                    img = cv2.imread(img_path)
                    img_list.append(img)

            if(len(img_list) != len(folder_path)):
                continue

            fig, axs = plt.subplots(1, len(img_list), figsize=(15, 5))

            for i in range(len(img_list)):
                img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB)
                axs[i].imshow(img_list[i])
                axs[i].set_title(f"{filename} - v1.{i}")
                axs[i].axis('off')

            plt.subplots_adjust(wspace=0.1)

            plt.savefig(os.path.join(concat_folder, filename), dpi=800)
            plt.close()
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

if __name__ == '__main__':
    main_mip()
    # concat_images()



