import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.spatial import distance
import seaborn as sns
from scipy.ndimage import distance_transform_edt
from skimage import exposure
from tqdm import tqdm

from scipy.stats import gaussian_kde
from skimage import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import skimage
import pandas as pd

# from nnUNet.scripts.adaptive_gamma import down_sample
from skimage.transform import resize
import cv2
from scipy.ndimage import label

from nnUNet.scripts.resolution_unifier import xy_resolution
from nnUNet.scripts.test2 import MAX_PROCESSES
from cc3d import connected_components
import tifffile
from joblib import Parallel, delayed
# gaussian_filter1d
from scipy.ndimage import gaussian_filter1d

MAX_PROCESSERS = 16




# label_list = ['Raw', 'Adaptive_gamma', 'Global_gamma', 'Equalization']
# label_list = ['raw', 'diff_gamma_05_1', 'equalize', 'gamma_05', 'truncated_gamma_05', 'truncated_diff_gamma_05']
label_list = ['raw', 'diff_gamma_05_1', 'equalize', 'gamma_05', 'truncated_gamma_05', 'truncated_diff_gamma_05']
current_label = label_list
# 指定两个文件夹路径
folder1 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset175_14k_hb_neuron_aug_pure/imagesTr'
folder2 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/imagesTr'
label_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/labelsTr"
save_dir = r"/data/kfchen/trace_ws/gamma_trans_test/metrics_result"
# save_dir = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/full_com_gamma_0.7'
# mip_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/full_com_gamma_0.7/mip"
max_dist_from_soma = 2000

name_mapping_df = pd.read_csv(os.path.join("/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma", "name_mapping.csv"))
neuron_info_df = pd.read_csv("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv", encoding='gbk')


def fig_1(hist_total_list):
    col, row = 1, len(label_list)
    plt.rcParams['figure.dpi'] = 800
    fig, axes = plt.subplots(row, col, figsize=(col * 4, row * 1))

    plt.title('')
    plt.xlabel('Voxel value')
    # 设置y轴标签位置
    plt.ylabel('Frequency')
    handles = []
    labels = []

    for i in range(len(label_list)):
        # plt.subplot(row, col, i + 1)
        ax = axes[i]
        sns.histplot(x=bins, weights=hist_total_list[i] / np.sum(hist_total_list[0]), color=color_list[i],
                     label=mapped_label[i],
                     alpha=0.1, kde=True, bins=256, ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if (i != len(label_list) - 1):
            # 关闭数字而保留刻度
            ax.set_xticklabels([])
            ax.set_xlabel('')
        ax.set_ylabel('')
        # ax.set_ylim(0, np.max(hist_total_list[0]) / np.sum(hist_total_list[0]))
        ax.set_xlim(0, 255)
        handles.append(plt.Line2D([0], [0], color=color_list[i], lw=4))
        labels.append(mapped_label[i])

        middle_value_ratio = np.sum(hist_total_list[i][int(middle_range[0] * 255):int(middle_range[1] * 255)]) / np.sum(
            hist_total_list[i])
        ax.text(0.5, 0.5, f'* {middle_value_ratio * 100:.2f}%', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

    print(handles, mapped_label)
    plt.legend(handles=handles, labels=labels, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.9), ncol=1, fontsize=14)

    from matplotlib.font_manager import FontProperties
    for text in plt.gca().get_legend().get_texts():
        if 'DAGT' in text.get_text():
            text.set_fontproperties(FontProperties(weight='bold', size=14))

    # 全局y轴标签
    fig.text(0.01, 0.6, 'Normalized Frequency', va='center', rotation='vertical')
    # fig.text(0.08, 0.008, "   *     Percentage of medium-intensity voxels", ha='left')
    plt.subplots_adjust(bottom=0.6, left=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"/data/kfchen/trace_ws/gamma_trans_test/fig1.png")
    plt.close()


def fig_2(i_dist_total_list, config=1, fig_file="fig2.png"):
    if(config==1):
        plt.figure(figsize=(5, 3))
    else:
        plt.figure(figsize=(4, 3))
    # sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum1/np.max(mi_hist_sum1), color='black', label='origin')
    # sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum2/np.max(mi_hist_sum2), color='red', label='gamma')
    i_dist_total_list = [i_dist_total_list[i][1:] for i in range(len(label_list))]
    # max non zero arg
    max_dist = 0
    for i in range(len(label_list)):
        for j in range(len(i_dist_total_list[i])):
            if (i_dist_total_list[i][j] > 0):
                max_dist = max(max_dist, j)
    i_dist_total_list = [i_dist_total_list[i][1:max_dist] for i in range(len(label_list))]
    if (config == 1):
        # full
        plt.xlim(0, max_dist)
        plt.ylim(0, 255)
    elif (config == 2):
        # low
        plt.xlim(0, 15)
        plt.ylim(0, 255)
    # middle
    elif (config == 3):
        plt.xlim(15, 150)
        plt.ylim(0, 40)
    for i in range(len(label_list)):
        sns.lineplot(x=np.linspace(1, max_dist, max_dist - 1), y=i_dist_total_list[i] / np.sum(hist_total_list[0]),
                     color=color_list[i], label=mapped_label[i])
        print(i_dist_total_list[i] / np.sum(hist_total_list[0]))
    plt.title('')
    plt.xlabel('Dist. from Soma (μm)', fontsize=14)
    plt.ylabel("Mean voxel value\nin spherical shell", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend().set_visible(False)
    # plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    # plt.subplots_adjust(bottom=0.2)

    # 关闭上面和右边的边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 二值化 binarize
    fig = plt.gcf()  # 获取当前图形对象
    fig.tight_layout()
    plt.savefig(r"/data/kfchen/trace_ws/gamma_trans_test/" + fig_file)
    plt.close()


def fig_2_norm(i_dist_total_list, config=1, fig_file="fig2.png"):
    print("shape", np.array(i_dist_total_list).shape)
    print(i_dist_total_list[0])
    if(config==1):
        plt.figure(figsize=(5, 3), dpi=300)
    else:
        plt.figure(figsize=(4, 3), dpi=300)
    # sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum1/np.max(mi_hist_sum1), color='black', label='origin')
    # sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum2/np.max(mi_hist_sum2), color='red', label='gamma')

    bins = 100

    if (config == 1):
        # full
        plt.xlim(0.01, 1)
        plt.ylim(0, 255)
    elif (config == 2):
        # low
        plt.xlim(0.01, 0.1)
        plt.ylim(0, 255)
    # middle
    elif (config == 3):
        plt.xlim(0.1, 0.3)
        plt.ylim(0, 80)
    x_ = np.linspace(0, 1, bins)
    for i in range(len(label_list)):
        sns.lineplot(x=x_, y=i_dist_total_list[i] / np.sum(hist_total_list[0]),
                     color=color_list[i], label=mapped_label[i])
        # print(i_dist_total_list[i] / np.sum(hist_total_list[0]))
    plt.title('')
    plt.xlabel('Norm. dist. to soma', fontsize=14)
    plt.ylabel("Mean voxel value\nin spherical shell", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend().set_visible(False)
    # plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    # plt.subplots_adjust(bottom=0.2)

    # 关闭上面和右边的边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 二值化 binarize
    fig = plt.gcf()  # 获取当前图形对象
    fig.tight_layout()
    plt.savefig(r"/data/kfchen/trace_ws/gamma_trans_test/" + fig_file)
    plt.close()

def fig_3(bia_size_total_list, config=1, fig_file="fig3_1.png"):
    if(config==1):
        plt.figure(figsize=(5, 3))
    else:
        plt.figure(figsize=(3, 3))
    # max non zero arg
    max_dist = 0
    for i in range(len(label_list)):
        for j in range(len(bia_size_total_list[i])):
            if (bia_size_total_list[i][j] > 0):
                max_dist = max(max_dist, j)
    bia_size_total_list = [bia_size_total_list[i][1:max_dist] for i in range(len(label_list))]
    if (config == 1):
        plt.xlim(0, 251)
    else:
        plt.xlim(150, 250)
        plt.ylim(25, 50)
    for i in range(len(label_list)):
        sns.lineplot(x=np.linspace(1, max_dist, max_dist - 1), y=bia_size_total_list[i] / np.sum(hist_total_list[0]),
                     color=color_list[i], label=mapped_label[i])
    plt.title('')
    plt.xlabel('Binarize threshold', fontsize=14)
    plt.ylabel('Soma block size (μm³)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend().set_visible(False)
    # plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    # plt.subplots_adjust(bottom=0.2)

    # 关闭上面和右边的边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 二值化 binarize
    fig = plt.gcf()  # 获取当前图形对象
    fig.tight_layout()
    plt.savefig(r"/data/kfchen/trace_ws/gamma_trans_test/" + fig_file)
    plt.close()


def find_resolution(filename):
    nnunet_name = os.path.basename(filename).replace('_0000.tif', '')
    df = name_mapping_df
    id = df[df['nnunet_name'] == nnunet_name]['ID'].values[0]
    # print(id)
    df = neuron_info_df
    xy_resolution = df.loc[df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0]
    return xy_resolution
    # for i in range(len(df)):
    #     if int(df.iloc[i, 0]) == id:
    #         return df.iloc[i, 43]
    # return None

def get_origin_img_size(file_name):
    # print(file_name)
    df = name_mapping_df
    full_name = os.path.basename(file_name).replace('_0000.tif', '')
    # full_name = int(full_name.split('_')[0])
    # print(full_name)
    # print(full_name)
    img_size = df[df['nnunet_name'] == full_name]['img_size'].values[0]
    img_size = img_size.split(',')
    x_limit, y_limit, z_limit = img_size[2], img_size[1], img_size[0]
    x_limit, y_limit, z_limit = "".join(filter(str.isdigit, x_limit)), \
        "".join(filter(str.isdigit, y_limit)), \
        "".join(filter(str.isdigit, z_limit))
    origin_size = (int(z_limit), int(y_limit), int(x_limit))
    return origin_size

def quick_soma_from_seg(seg):
    # 进行距离变换
    distance_transformed = distance_transform_edt(seg)

    # 找到最大值及其所在的体素坐标
    # max_distance = np.max(distance_transformed)
    max_coord = np.unravel_index(np.argmax(distance_transformed), distance_transformed.shape)
    return max_coord

def calc_dt_from_soma(img, soma_coord):
    # 创建一个与图像相同大小的空白掩码，并在起始点设为1
    mask = np.zeros_like(img, dtype=bool)
    mask[soma_coord] = True

    # 计算从起始点到图像中每个点的欧几里得距离
    distance_map = distance_transform_edt(~mask)

    # 计算不同距离范围内的平均体素强度
    max_distance = int(np.ceil(distance_map.max()))
    mean_intensity = []

    for r in range(1, max_distance + 1):
        # 找到距离在 r - 1 和 r 之间的体素
        mask_r = (distance_map >= r - 1) & (distance_map < r)

        # 如果该距离范围内有体素，计算平均强度
        if np.any(mask_r):
            mean_intensity.append(np.mean(img[mask_r]))
        else:
            mean_intensity.append(0)

    bins = np.linspace(0, max_dist_from_soma, max_dist_from_soma)
    binned_intensity = []
    binned_intensity.append(0)

    for i in range(1, len(bins)):
        if(i > len(mean_intensity)):
            binned_intensity.append(0)
            continue
        bin_mask = (distance_map >= bins[i - 1]) & (distance_map < bins[i])
        if np.any(bin_mask):
            binned_intensity.append(np.mean(img[bin_mask]))
        else:
            binned_intensity.append(0)

    # print(len(binned_intensity))
    return binned_intensity

def calc_binarize_threshold_soma_size_relation(img, soma_coord):
    binarize_threshold = range(0, 256)
    soma_crop_size_list = []

    def get_soma_cc(binary_image, target_point):
        # labeled_image, num_features = label(binary_image)
        # target_label = labeled_image[target_point]
        # connected_region = np.argwhere(labeled_image == target_label)
        # return connected_region
        # use cc3d
        if(binary_image[target_point] == 0):
            binary_image[target_point] = 1

        labeled_image = connected_components(binary_image)
        target_label = labeled_image[target_point]
        connected_region = np.argwhere(labeled_image == target_label)
        return connected_region


    def find_bounding_cube(connected_region):
        min_coords = connected_region.min(axis=0)
        max_coords = connected_region.max(axis=0)
        z_span, y_span, x_span = max_coords - min_coords
        cube_size = z_span * y_span * x_span
        return cube_size

    for threshold in binarize_threshold:
        binary_image = img > threshold
        soma_cc = get_soma_cc(binary_image, soma_coord)
        soma_crop_size = find_bounding_cube(soma_cc)
        soma_crop_size_list.append(soma_crop_size)
    return soma_crop_size_list

def get_gamma_img(img):
    img = exposure.adjust_gamma(img, gamma=0.75)
    # img = exposure.equalize_adapthist(img, clip_limit=0.02, nbins=256)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def get_equalize_img(img):
    img = exposure.equalize_adapthist(img, clip_limit=0.02, nbins=256)
    img = (img - img.min()) / (img.max() - img.min())
    return img



def get_total_list():
    # 找到图像id
    test_list_file = "/data/kfchen/trace_ws/paper_trace_result/test_list_with_gs.csv"
    test_list = pd.read_csv(test_list_file)["id"].tolist()
    test_list = [int(f) for f in test_list]
    name_mapping = {}


    # 找到对应的原始图像
    ori_tif_dir = "/data/kfchen/trace_ws/de_flu_test/14k_tif"
    down_sample_tif_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_tif"
    down_sample_res = 1
    for f in os.listdir(ori_tif_dir):
        if f.endswith(".tif"):
            name_mapping[int(f.split('_')[0])] = f
    # os.remove(down_sample_tif_dir)
    if(not os.path.exists(down_sample_tif_dir)):
        os.makedirs(down_sample_tif_dir)
        tif_files = [f for f in os.listdir(ori_tif_dir) if f.endswith(".tif")]
        tif_files = [f for f in tif_files if int(f.split('_')[0]) in test_list]

        def copy_and_down_sample_tif(f):
            img = io.imread(os.path.join(ori_tif_dir, f))
            img = np.array(img).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            # resize
            if(not down_sample_res == 1):
                img = resize(img, (int(img.shape[0] / down_sample_res), int(img.shape[1] / down_sample_res), int(img.shape[2] / down_sample_res)), order=2, anti_aliasing=False)
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            io.imsave(os.path.join(down_sample_tif_dir, f), img)

        with ThreadPoolExecutor(max_workers=MAX_PROCESSERS) as executor:
            futures = {}
            for f in tif_files:
                futures[executor.submit(copy_and_down_sample_tif, f)] = f
            for future in as_completed(futures):
                pass

    # 准备下采样swc文件
    swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/2_flip_after_sort"
    down_sample_swc_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_swc"
    if(not os.path.exists(down_sample_swc_dir)):
        os.makedirs(down_sample_swc_dir)
        swc_files = [f for f in os.listdir(swc_dir) if f.endswith(".swc")]
        swc_files = [f for f in swc_files if int(f.split('.')[0]) in test_list]
        for swc_file in swc_files:
            with open(os.path.join(swc_dir, swc_file), 'r') as f:
                lines = f.readlines()
            with open(os.path.join(down_sample_swc_dir, name_mapping[int(swc_file.split('.')[0])]).replace(".tif", ".swc"), 'w') as f:
                for line in lines:
                    if line.startswith("#"):
                        f.write(line)
                    else:
                        line = line.split(' ')
                        line[2] = str(float(line[2]) / down_sample_res)
                        line[3] = str(float(line[3]) / down_sample_res)
                        line[4] = str(float(line[4]) / down_sample_res)
                        f.write(' '.join(line))

    def get_soma_for_tif_file(img_file):
        swc_file = img_file.replace(".tif", ".swc")
        with open(os.path.join(down_sample_swc_dir, swc_file), 'r') as f:
            lines = f.readlines()
        soma = None
        for line in lines:
            if line.startswith("#"):
                continue
            line = line.split(' ')
            if int(line[1]) == 1:
                soma = (int(float(line[2])), int(float(line[3])), int(float(line[4])))
                break

        soma = [int(f) for f in soma]
        soma = (soma[2], soma[1], soma[0])
        return soma
    # # 检查soma位置
    # img_files = [f for f in os.listdir(down_sample_tif_dir) if f.endswith(".tif")]
    # for img_file in img_files:
    #     img = io.imread(os.path.join(down_sample_tif_dir, img_file))
    #     soma = get_soma_for_tif_file(img_file)
    #     print(img[soma[0], soma[1], soma[2]], soma, img.shape)

    neuron_info_df = pd.read_csv("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv", encoding='gbk')
    from nnUNet.scripts.Elimination_of_fluorescence import diffusion_adaptive_gamma
    def generate_diff_adaptive_gamma_img(img_file, adaptive_gamma_dir, gamma_map_dir):
        img = io.imread(os.path.join(down_sample_tif_dir, img_file))
        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())

        # filename = int(filename.split('_')[0].split('.')[0])
        # return df[df['Cell ID'] == filename]['xy拍摄分辨率(*10e-3μm/px)'].values[0]
        xy_resolution = neuron_info_df[neuron_info_df['Cell ID'] == int(img_file.split('_')[0].split('.')[0])]['xy拍摄分辨率(*10e-3μm/px)'].values[0]
        # print(xy_resolution)

        img, gamma_map = diffusion_adaptive_gamma(img, xy_resolution)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        gamma_map = ((gamma_map - gamma_map.min()) / (gamma_map.max() - gamma_map.min()) * 255).astype(np.uint8)
        io.imsave(os.path.join(adaptive_gamma_dir, img_file), img)
        io.imsave(os.path.join(gamma_map_dir, img_file), gamma_map)

    diff_gamma_tif_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_diff_gamma"
    diff_gamma_map_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_diff_gamma_map"
    if(not os.path.exists(diff_gamma_tif_dir)):
        os.makedirs(diff_gamma_tif_dir)
        os.makedirs(diff_gamma_map_dir)
        img_files = [f for f in os.listdir(down_sample_tif_dir) if f.endswith(".tif")]
        with ThreadPoolExecutor(max_workers=MAX_PROCESSERS) as executor:
            futures = {}
            for f in img_files:
                futures[executor.submit(generate_diff_adaptive_gamma_img, f, diff_gamma_tif_dir, diff_gamma_map_dir)] = f
            for future in as_completed(futures):
                pass
    # 直方图均衡化
    def generate_equalize_img(img_file, equalize_dir):
        img = io.imread(os.path.join(down_sample_tif_dir, img_file))
        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = exposure.equalize_adapthist(img, clip_limit=0.02, nbins=256)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        io.imsave(os.path.join(equalize_dir, img_file), img)
    equalize_tif_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_equalize"
    if(not os.path.exists(equalize_tif_dir)):
        os.makedirs(equalize_tif_dir)
        img_files = [f for f in os.listdir(down_sample_tif_dir) if f.endswith(".tif")]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for f in img_files:
                futures[executor.submit(generate_equalize_img, f, equalize_tif_dir)] = f
            for future in as_completed(futures):
                pass


    def generate_gamma_img(img_file, gamma_dir):
        img = io.imread(os.path.join(down_sample_tif_dir, img_file))
        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = exposure.adjust_gamma(img, gamma=0.5)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        io.imsave(os.path.join(gamma_dir, img_file), img)
    gamma_tif_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_gamma_05"
    if(not os.path.exists(gamma_tif_dir)):
        os.makedirs(gamma_tif_dir)
        img_files = [f for f in os.listdir(down_sample_tif_dir) if f.endswith(".tif")]
        with ThreadPoolExecutor(max_workers=MAX_PROCESSERS) as executor:
            futures = {}
            for f in img_files:
                futures[executor.submit(generate_gamma_img, f, gamma_tif_dir)] = f
            for future in as_completed(futures):
                pass

    from nnUNet.scripts.Elimination_of_fluorescence import truncated_gamma
    def generate_truncated_gamma_img(img_file, truncated_gamma_dir):
        img = io.imread(os.path.join(down_sample_tif_dir, img_file))
        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        gamma_map = np.ones_like(img) * 0.5
        img = truncated_gamma(img, gamma_map, (0.5, 0.5))
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        io.imsave(os.path.join(truncated_gamma_dir, img_file), img)
    truncated_gamma_tif_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_truncated_gamma_05"
    if(not os.path.exists(truncated_gamma_tif_dir)):
        os.makedirs(truncated_gamma_tif_dir)
        img_files = [f for f in os.listdir(down_sample_tif_dir) if f.endswith(".tif")]
        with ThreadPoolExecutor(max_workers=MAX_PROCESSERS) as executor:
            futures = {}
            for f in img_files:
                futures[executor.submit(generate_truncated_gamma_img, f, truncated_gamma_tif_dir)] = f
            for future in as_completed(futures):
                pass

    from nnUNet.scripts.Elimination_of_fluorescence import truncated_diff_gamma
    def generate_truncated_diff_gamma_img(img_file, truncated_diff_gamma_dir):
        img = io.imread(os.path.join(down_sample_tif_dir, img_file))
        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())

        # filename = int(filename.split('_')[0].split('.')[0])
        # return df[df['Cell ID'] == filename]['xy拍摄分辨率(*10e-3μm/px)'].values[0]
        xy_resolution = neuron_info_df[neuron_info_df['Cell ID'] == int(img_file.split('_')[0].split('.')[0])][
            'xy拍摄分辨率(*10e-3μm/px)'].values[0]
        # print(xy_resolution)

        img = truncated_diff_gamma(img, xy_resolution)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        io.imsave(os.path.join(truncated_diff_gamma_dir, img_file), img)

    truncated_diff_gamma_tif_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_truncated_diff_gamma_05"
    if(not os.path.exists(truncated_diff_gamma_tif_dir)):
        os.makedirs(truncated_diff_gamma_tif_dir)
        img_files = [f for f in os.listdir(down_sample_tif_dir) if f.endswith(".tif")]
        with ThreadPoolExecutor(max_workers=MAX_PROCESSERS) as executor:
            futures = {}
            for f in img_files:
                futures[executor.submit(generate_truncated_diff_gamma_img, f, truncated_diff_gamma_tif_dir)] = f
            for future in as_completed(futures):
                pass

    # compare_dirs = [down_sample_tif_dir, diff_gamma_tif_dir, equalize_tif_dir, gamma_tif_dir, diff_gamma_map_dir,
    #                 truncated_gamma_tif_dir, truncated_diff_gamma_tif_dir]
    # mip_label = ['raw', 'diff_gamma_05_1', 'equalize', 'gamma_05', 'diff_gamma_map', 'truncated_gamma_05', 'truncated_diff_gamma_05']
    compare_dirs = [down_sample_tif_dir, diff_gamma_tif_dir, truncated_gamma_tif_dir, gamma_tif_dir, equalize_tif_dir]
    mip_label = ['raw', 'diff_gamma_05_1', 'truncated_gamma_05', 'gamma_05', 'equalize']
    mip_compare_dir = "/data/kfchen/trace_ws/gamma_trans_test/mip_compare"
    if(not os.path.exists(mip_compare_dir)):
        os.makedirs(mip_compare_dir)
        img_files = [f for f in os.listdir(compare_dirs[-1]) if f.endswith(".tif")]
        for img_file in img_files:
            compare_img_files = [os.path.join(f, img_file) for f in compare_dirs]
            compare_imgs = [io.imread(f) for f in compare_img_files]
            mip_imgs = [np.max(f, axis=0) for f in compare_imgs]
            # 给每个mip图像加上标签
            mip_imgs = [cv2.putText(f, mip_label[i], (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA) for i, f in enumerate(mip_imgs)]
            # 加0.05的百边
            mip_imgs = [cv2.copyMakeBorder(f, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for f in mip_imgs]
            mip_img = np.concatenate(mip_imgs, axis=1)
            io.imsave(os.path.join(mip_compare_dir, img_file.replace('.tif', '.png')), mip_img)


    tif_img_files = [f for f in os.listdir(down_sample_tif_dir) if f.endswith(".tif")]
    hist_total_list = [np.zeros(256) for i in range(len(current_label))]
    i_dist_total_list = [np.zeros(max_dist_from_soma) for i in range(len(current_label))]
    bia_size_total_list = [np.zeros(256) for i in range(len(current_label))]
    norm_i_dist_total_list = [np.zeros(100) for i in range(len(current_label))]

    pbar = tqdm(total=len(tif_img_files))
    foloder_list = [down_sample_tif_dir, diff_gamma_tif_dir, equalize_tif_dir, gamma_tif_dir, truncated_gamma_tif_dir, truncated_diff_gamma_tif_dir]

    temp_resized_tif_dir = r"/data/kfchen/trace_ws/gamma_trans_test/resized_tif"

    def current_task(file_name, folder_list):
        down_sample_r = 2
        origin_shape = io.imread(os.path.join(folder_list[0], file_name)).shape
        # print(origin_shape)
        xy_resolution = neuron_info_df[neuron_info_df['Cell ID'] == int(file_name.split('_')[0].split('.')[0])][
            'xy拍摄分辨率(*10e-3μm/px)'].values[0]
        soma_coord = get_soma_for_tif_file(file_name)
        soma_coord = [int(f * xy_resolution / 1000 / down_sample_r) for f in soma_coord]
        soma_coord = (soma_coord[0], soma_coord[1], soma_coord[2])

        img_list = [] # 存的是这一个图像经过不同方法处理后的图像

        for f in folder_list:
            temp_current_image_file = os.path.join(temp_resized_tif_dir, file_name.replace(".tif", "_") + os.path.basename(f) + str(down_sample_r) + "_.tif")
            if(not os.path.exists(temp_current_image_file)):
                current_image = io.imread(os.path.join(f, file_name))
                current_image = current_image.astype(np.float32)
                current_image = (current_image - current_image.min()) / (current_image.max() - current_image.min())
                out_shape = (int(origin_shape[0] * xy_resolution / 1000 / down_sample_r), int(origin_shape[1] * xy_resolution / 1000 / down_sample_r), int(origin_shape[2] * xy_resolution / 1000 / down_sample_r))
                current_image = resize(current_image, out_shape, order=2, anti_aliasing=False)
                current_image = (current_image - current_image.min()) / (current_image.max() - current_image.min())
                current_image = (current_image * 255).astype("uint8")
                img_list.append(current_image)
                io.imsave(temp_current_image_file, current_image)
            else:
                current_image = io.imread(temp_current_image_file)
                current_image = current_image.astype(np.float32)
                current_image = (current_image - current_image.min()) / (current_image.max() - current_image.min()) * 255
                current_image = current_image.astype("uint8")
                img_list.append(current_image)
            # print(current_image[soma_coord[0], soma_coord[1], soma_coord[2]])
            # print(soma_coord, current_image.shape)
        #
        # img_list = [io.imread(os.path.join(f, file_name)) for f in folder_list]
        # img_list = [f.astype("float32") for f in img_list]
        # img_list = [(f - f.min()) / (f.max() - f.min()) for f in img_list]
        #
        # img_shape = [int(origin_shape[0]), int(origin_shape[1] * xy_resolution / 1000),
        #              int(origin_shape[2] * xy_resolution / 1000)]
        # img_shape = [int(f / down_sample_r) for f in img_shape]
        # img_list = [skimage.transform.resize(current_img, img_shape, order=2, anti_aliasing=False) for current_img in
        #             img_list]
        # img_list = [(f * 255).astype(np.uint8) for f in img_list]





        # 更新直方图和平均强度值
        # 这个list记录的是这一个图像，在几种方法处理后的各项指标
        local_hist_list = [np.zeros(256) for _ in range(len(current_label))]
        local_i_dist_sum = [np.zeros(max_dist_from_soma) for _ in range(len(current_label))]
        local_norm_i_dist_sum = [np.zeros(100) for _ in range(len(current_label))]
        local_bia_size_list = [np.zeros(256) for _ in range(len(current_label))]

        for i, img in enumerate(img_list):
            print(img.shape)
            current_hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
            local_hist_list[i] += current_hist
            print("histogram done")

            temp_current_mean_intensity_file = os.path.join(temp_resized_tif_dir, file_name.replace(".tif", "_") + os.path.basename(folder_list[i]) + str(down_sample_r) + "_i_dist.npy")
            if(not os.path.exists(temp_current_mean_intensity_file)):
                current_mean_intensity = calc_dt_from_soma(img, soma_coord)
                np.save(temp_current_mean_intensity_file, current_mean_intensity)
            else:
                current_mean_intensity = np.load(temp_current_mean_intensity_file)
            local_i_dist_sum[i] += current_mean_intensity

            # 归一化结果
            temp_current_norm_i_dist_sum_file = os.path.join(temp_resized_tif_dir, file_name.replace(".tif", "_") + os.path.basename(folder_list[i]) + str(down_sample_r) + "_norm_i_dist.npy")
            if(not os.path.exists(temp_current_norm_i_dist_sum_file)):
                current_mean_intensity = current_mean_intensity[1:]
                max_dist = 0
                for j in range(len(current_mean_intensity)):
                    if(current_mean_intensity[j] > 0):
                        max_dist = j
                print(f"max_dist: {max_dist}")
                current_norm_i_dist_sum = np.zeros(100)
                for j in range(100):
                    mi = np.mean(current_mean_intensity[int(j / 100 * max_dist):int((j + 1) / 100 * max_dist)])
                    # is nan
                    if(not np.isnan(mi)):
                        current_norm_i_dist_sum[j] = mi
                    else:
                        mi = 0
                        print(f"dist: {int(j / 100 * max_dist)}-{int((j + 1) / 100 * max_dist)}", f"mean: {mi}")
                np.save(temp_current_norm_i_dist_sum_file, current_norm_i_dist_sum)
            else:
                current_norm_i_dist_sum = np.load(temp_current_norm_i_dist_sum_file)
            local_norm_i_dist_sum[i] += current_norm_i_dist_sum
            print("distance transform done")

            temp_current_bia_size_file = os.path.join(temp_resized_tif_dir, file_name.replace(".tif", "_") + os.path.basename(folder_list[i]) + str(down_sample_r) + "_bia_size.npy")
            if(not os.path.exists(temp_current_bia_size_file)):
                current_bia_size = calc_binarize_threshold_soma_size_relation(img, soma_coord)
                np.save(temp_current_bia_size_file, current_bia_size)
            else:
                current_bia_size = np.load(temp_current_bia_size_file)
            local_bia_size_list[i] += current_bia_size
            print("bia size done")

        # print(local_hist_list[0][:10], local_i_dist_sum[0][:10])
        return local_hist_list, local_i_dist_sum, local_bia_size_list, local_norm_i_dist_sum

    # for file_name in tif_img_files:
    #     local_hist_list, local_i_dist_list, local_bia_size_list = current_task(file_name, foloder_list)
    #     pbar.update(1)
    # 多线程
    pbar = tqdm(total=len(tif_img_files))
    with ThreadPoolExecutor() as executor:
        futures = {}
        for file_name in tif_img_files:
            futures[executor.submit(current_task, file_name, foloder_list)] = file_name
        for future in as_completed(futures):
            pbar.update(1)



    for file_name in tif_img_files:
        local_hist_list, local_i_dist_list, local_bia_size_list, local_norm_i_dist_sum = current_task(file_name, foloder_list)

        # # 更新全局变量
        for i in range(len(current_label)):
            hist_total_list[i] += local_hist_list[i]
            i_dist_total_list[i] += local_i_dist_list[i]
            bia_size_total_list[i] += local_bia_size_list[i]
            norm_i_dist_total_list[i] += local_norm_i_dist_sum[i]


    pbar.close()


    for i in range(len(current_label)):
        if(not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, 'hist_total_' + current_label[i] + '.npy'), hist_total_list[i])
        np.save(os.path.join(save_dir, 'i_dist_total_' + current_label[i] + '.npy'), i_dist_total_list[i])
        np.save(os.path.join(save_dir, 'bia_size_total_' + current_label[i] + '.npy'), bia_size_total_list[i])
        np.save(os.path.join(save_dir, 'norm_i_dist_total_' + current_label[i] + '.npy'), norm_i_dist_total_list[i])

def prepare_down_sample_rescaled_swc():
    source_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab"
    target_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_down_sample_2"
    if(not os.path.exists(target_swc_dir)):
        os.makedirs(target_swc_dir)
        swc_files = [f for f in os.listdir(source_swc_dir) if f.endswith(".swc")]
        for swc_file in swc_files:
            with open(os.path.join(source_swc_dir, swc_file), 'r') as f:
                lines = f.readlines()
            with open(os.path.join(target_swc_dir, swc_file), 'w') as f:
                for line in lines:
                    if line.startswith("#"):
                        f.write(line)
                    else:
                        line = line.split(' ')
                        line[2] = str(float(line[2]) / 2)
                        line[3] = str(float(line[3]) / 2)
                        line[4] = str(float(line[4]) / 2)
                        f.write(' '.join(line))

def traverse_from_soma(swc_df, img):
    # 找到soma节点（通常是type == 1）
    soma_node = swc_df[swc_df['type'] == 1].iloc[0]
    soma_id = soma_node['id']

    # 创建一个字典来存储从每个节点到其子节点的连接关系
    tree = {}
    for _, row in swc_df.iterrows():
        if row['parent'] != -1:  # parent == -1表示没有父节点（根节点）
            if row['parent'] not in tree:
                tree[row['parent']] = []
            tree[row['parent']].append(row['id'])

    # 存储每个节点到soma的路径距离和直线距离
    distance_to_soma = {soma_id: 0.0}  # soma到自己的距离为0
    straight_line_distance = {soma_id: 0.0}  # soma到自己的直线距离为0
    img_value = {soma_id: img[int(soma_node['z']), int(soma_node['y']), int(soma_node['x'])]}
    visited = set()  # 记录已访问的节点

    # 计算两个节点之间的欧氏距离
    def euclidean_distance(p1, p2):
        return np.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2 + (p1['z'] - p2['z']) ** 2)

    # 深度优先搜索DFS，从soma开始遍历
    def dfs(node_id, current_distance, current_straight_distance):
        # 遍历该节点的所有子节点
        if node_id in visited:
            return

        visited.add(node_id)

        # 获取当前节点的信息
        current_node = swc_df[swc_df['id'] == node_id].iloc[0]

        # 记录当前节点到soma的路径距离（路径总和）和直线距离
        distance_to_soma[node_id] = current_distance
        straight_line_distance[node_id] = current_straight_distance
        z, y, x = int(current_node['z']), int(current_node['y']), int(current_node['x'])
        z, y, x = min(max(z, 0), img.shape[0]-1), min(max(y, 0), img.shape[1]-1), min(max(x, 0), img.shape[2]-1)
        img_value[node_id] = img[z, y, x]

        # 遍历所有子节点
        if node_id in tree:
            for child_id in tree[node_id]:
                # 计算从当前节点到子节点的直线距离
                child_node = swc_df[swc_df['id'] == child_id].iloc[0]
                edge_distance = euclidean_distance(current_node, child_node)
                # 递归调用DFS，累加路径距离和直线距离
                dfs(child_id, current_distance + edge_distance, euclidean_distance(soma_node, child_node))

    # 从soma节点开始遍历
    dfs(soma_id, 0.0, 0.0)

    swc_df['path_dist'] = np.nan
    swc_df['euclidean_dist'] = np.nan
    swc_df['image_intensity'] = np.nan
    for node_id in distance_to_soma:
        swc_df.loc[swc_df['id'] == node_id, 'path_dist'] = distance_to_soma[node_id]
        swc_df.loc[swc_df['id'] == node_id, 'euclidean_dist'] = straight_line_distance[node_id]
        swc_df.loc[swc_df['id'] == node_id, 'image_intensity'] = img_value[node_id]

    # return distance_to_soma, straight_line_distance, img_value
    return swc_df

def calc_skel_intensity_distribution_file(img_file, swc_file, save_file):
    # print(save_file)
    if os.path.exists(save_file):
        swc_df = pd.read_csv(save_file)
        return swc_df['path_dist'], swc_df['image_intensity']

    img = tifffile.imread(img_file).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    # img = np.flip(img, axis=1)

    swc_df = pd.read_csv(swc_file, sep=' ', header=None, comment='#',
                     names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'],
                     dtype={'id': int, 'type': int, 'x': float, 'y': float, 'z': float, 'radius': float, 'parent': int})

    try:
        swc_df = traverse_from_soma(swc_df, img)
    except:
        print(f"Error in {swc_file}")
        swc_df['path_dist'] = np.nan
        swc_df['image_intensity'] = np.nan

    swc_df.to_csv(save_file, index=False)
    # print(swc_df['path_dist'][:5])
    # print(swc_df['image_intensity'][:5])

    return swc_df['path_dist'], swc_df['image_intensity']

def calc_skel_intensity_distribution_dir(img_dir, swc_dir, save_dir):
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    # for img_file in img_files:
    #     swc_file = os.path.join(swc_dir, img_file.replace('.tif', '.swc'))
    #     save_file = os.path.join(save_dir, img_file.replace('.tif', '.csv'))
    #     calc_skel_intensity_distribution_file(os.path.join(img_dir, img_file), swc_file, save_file)
    Parallel(n_jobs=MAX_PROCESSERS)(
        delayed(calc_skel_intensity_distribution_file)(
            os.path.join(img_dir, img_file),
            os.path.join(swc_dir, img_file.replace('.tif', '.swc')),
            os.path.join(save_dir, img_file.replace('.tif', '.csv'))
        ) for img_file in tqdm(img_files)
    )

def calc_skel_intensity_distribution_for_trans():
    img_root = "/data/kfchen/trace_ws/gamma_trans_test"
    trans_methods = ['raw', 'diff_gamma', 'equalize', 'gamma_05', 'truncated_gamma_05']
    swc_dir = "/data/kfchen/trace_ws/gamma_trans_test/down_sample_242_swc"

    for trans_method in trans_methods:
        if(trans_method == 'raw'):
            img_dir = os.path.join(img_root, f"down_sample_242_tif")
        else:
            img_dir = os.path.join(img_root, f"down_sample_242_{trans_method}")
        save_dir = os.path.join("/data/kfchen/trace_ws/gamma_trans_test/temp_skel_instensity", trans_method)
        if(not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        calc_skel_intensity_distribution_dir(img_dir, swc_dir, save_dir)


def plt_fig1(num_bins=1000):
    trans_methods = ['raw', 'diff_gamma', 'truncated_gamma_05', 'gamma_05', 'equalize']
    result_root = "/data/kfchen/trace_ws/gamma_trans_test/temp_skel_instensity"
    result_dirs = [os.path.join(result_root, f) for f in trans_methods]

    path_dists = [[] for _ in range(len(trans_methods))]
    image_intensities = [[] for _ in range(len(trans_methods))]
    for i, result_dir in enumerate(result_dirs):
        result_files = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
        for result_file in result_files:
            df = pd.read_csv(os.path.join(result_dir, result_file))[['path_dist', 'image_intensity']]
            df = df.dropna()
            current_path_dist, current_image_intensity = df['path_dist'], df['image_intensity']
            current_path_dist = current_path_dist / np.max(current_path_dist) * num_bins
            current_path_dist = [int(i) for i in current_path_dist]
            current_image_intensity = (current_image_intensity - current_image_intensity.min()) / (
                        current_image_intensity.max() - current_image_intensity.min()) * 255
            if (len(current_image_intensity) == 0 or current_image_intensity[0] < 255 * 0.5):
                continue
            path_dists[i].extend(current_path_dist)
            image_intensities[i].extend(current_image_intensity)


    # hist
    # 设置清晰度
    plt.rcParams['figure.dpi'] = 300
    color_list = plt.cm.get_cmap('Set3').colors
    set2_colors = [(0, 0, 0), color_list[3], color_list[6], color_list[2], color_list[4]]
    plt.figure(figsize=(4, 3))

    mean_intensities = []
    for i in range(len(trans_methods)):
        df = pd.DataFrame({
            'path_dist': np.array(path_dists[i]).astype(np.float32) / num_bins,
            'image_intensity': image_intensities[i]
        })
        average_intensities = df.groupby('path_dist')['image_intensity'].mean().reset_index()
        mean_intensities.append(average_intensities)

    for i in range(len(mean_intensities)):
        current_x, current_y = mean_intensities[i]['path_dist'], mean_intensities[i]['image_intensity']
        # 平滑y
        current_y = gaussian_filter1d(current_y, sigma=2)
        plt.plot(current_x, current_y,
                 color=set2_colors[i], linewidth=1)

    plt.xlim(0, 1)
    plt.ylim(0, 255)

    plt.xlabel('Norm. path dist. to soma', fontsize=15)
    plt.ylabel('Voxel value', fontsize=15)
    # tick
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.title('Path Distance to Soma vs Image Intensity')
    # legend
    # plt.legend(['Dye-injection', 'Genetic labeling'], fontsize=12, frameon=False)
    # plt.legend(['Raw', 'Diffusion-based adaptive gamma', 'Histogram equalization', 'Global gamma', 'Truncated gamma'],
    #             fontsize=12, frameon=False)
    plt.tight_layout()
    # plt.colorbar(label='Image Intensity')
    # plt.show()
    plt.savefig("/data/kfchen/trace_ws/gamma_trans_test/temp_skel_instensity/Path_Distance_to_Soma_vs_Image_Intensity.png")
    plt.close()

# def update_pbar(pbar, value):
#     pbar.update(value)
#
#
# # 处理每个文件的函数
# def process_file(file_name, label_dir, folder1, folder2):
#     # origin_shape = get_origin_img_size(file_name)
#     origin_shape
#     xy_resolution = int(find_resolution(file_name))
#     img_shape = [int(origin_shape[0]), int(origin_shape[1] * xy_resolution / 1000),
#                  int(origin_shape[2] * xy_resolution / 1000)]
#     img_shape = [int(f / 2) for f in img_shape]
#
#
#
#     lab = io.imread(os.path.join(label_dir, file_name.replace("_0000.tif", ".tif")))
#     lab = skimage.transform.resize(lab, img_shape, order=0, anti_aliasing=False)
#     soma_coord = quick_soma_from_seg(lab)
#
#     img_list = [
#         io.imread(os.path.join(folder1, file_name)),
#         io.imread(os.path.join(folder2, file_name))
#     ]
#
#     img_list = [f.astype("float32") for f in img_list]
#     img_list = [(f - f.min()) / (f.max() - f.min()) for f in img_list]
#
#
#     img_list = [skimage.transform.resize(current_img, img_shape, order=2, anti_aliasing=False) for current_img in img_list]
#     # print(img_list[0].shape)
#
#     img_list.append(get_gamma_img(img_list[0]))
#     img_list.append(get_equalize_img(img_list[0]))
#
#     img_list = [(f * 255).astype(np.uint8) for f in img_list]
#
#     # 更新直方图和平均强度值
#     local_hist_list = [np.zeros(256) for _ in range(4)]
#     local_i_dist_sum = [np.zeros(max_dist_from_soma) for _ in range(4)]
#
#     for i, img in enumerate(img_list):
#         current_hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
#         local_hist_list[i] += current_hist
#
#         current_mean_intensity = calc_dt_from_soma(img, soma_coord)
#         local_i_dist_sum[i] += current_mean_intensity
#
#     return local_hist_list, local_i_dist_sum
#

# def process_files_in_parallel(common_files, label_dir, folder1, folder2):
#     pbar = tqdm(total=len(common_files))
#
#     # 线程池执行任务
#     with ThreadPoolExecutor() as executor:
#         futures = {}
#         for file_name in common_files:
#             futures[executor.submit(process_file, file_name, label_dir, folder1, folder2)] = file_name
#
#         # 等待所有线程完成
#         for future in as_completed(futures):
#             local_hist_sum, local_mi_hist_sum = future.result()
#
#             # 更新全局变量
#             for i in range(4):
#                 hist_sum_list[i] += local_hist_sum[i]
#                 mi_hist_sum_list[i] += local_mi_hist_sum[i]
#
#             # 更新进度条
#             update_pbar(pbar, 1)
#
#     pbar.close()


hist_total_list, i_dist_total_list, bia_size_total_list, norm_i_dist_total_list = [], [], [], []
# get_total_list()
prepare_down_sample_rescaled_swc()
calc_skel_intensity_distribution_for_trans()
plt_fig1()
exit()

label_list = ['raw', 'diff_gamma_05_1', 'truncated_gamma_05', 'gamma_05', 'equalize']
# label_map = {
#     'raw': 'Raw',
#     'diff_gamma_05_1': 'Diffusion-based adaptive gamma',
#     'truncated_gamma_05': 'Truncated gamma',
#     'gamma_05': 'Global gamma',
#     'equalize': 'Histogram equalization'
# }
mapped_label = ['Raw', 'DAGT', 'DTGT', 'Gamma trans.', 'Hist. equalization']
color_list = plt.cm.get_cmap('Set3').colors
color_list = [(0, 0, 0), color_list[3], color_list[6], color_list[2], color_list[4]]

bins = 100

for i in range(len(label_list)):
    hist_total_list.append(np.load(os.path.join(save_dir, 'hist_total_'  + label_list[i] + '.npy')))
    i_dist_total_list.append(np.load(os.path.join(save_dir, 'i_dist_total_' + label_list[i] + '.npy')))
    bia_size_total_list.append(np.load(os.path.join(save_dir, 'bia_size_total_' + label_list[i] + '.npy')))
    norm_i_dist_total_list.append(np.load(os.path.join(save_dir, 'norm_i_dist_total_' + label_list[i] + '.npy')))
# print(hist_total_list)
# print(i_dist_total_list)
# print(bia_size_total_list)







# 绘制两个文件夹的全图直方图和累计频率信息
bins = np.linspace(0, 256, 256)

middle_range = (0.25,  0.75) # 550
for i in range(len(label_list)):
    # print middle range
    print(f'{label_list[i]}: {np.sum(hist_total_list[i][int(middle_range[0]*128):int(middle_range[1]*128)]):.4f}')
    # print sum
    print(f'{label_list[i]}: {np.sum(hist_total_list[i]):.4f}', len(hist_total_list[i]))
# 0.01426



# fig_1(hist_total_list)

# fig_2(i_dist_total_list, 1, "fig2_1_full.png")
# fig_2(i_dist_total_list, 2, "fig2_2_low_focus.png")
# fig_2(i_dist_total_list, 3, "fig2_3_middle_focus.png")
fig_2_norm(norm_i_dist_total_list, 1, "fig2_1_full_norm.png")
fig_2_norm(norm_i_dist_total_list, 2, "fig2_2_low_focus_norm.png")
fig_2_norm(norm_i_dist_total_list, 3, "fig2_3_middle_focus_norm.png")

# fig_3(bia_size_total_list, 1, "fig3_1_full.png", )
# fig_3(bia_size_total_list, 2, "fig3_2_foucs.png")