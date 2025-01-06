import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
from skimage.transform import resize
from nnUNet.scripts.Elimination_of_fluorescence import find_best_sigma_map
import random

feature_names=['N_stem', 'Number of Bifurcatons', 'Number of Branches', 'Number of Tips',
                               'Overall Width', 'Overall Height', 'Overall Depth', 'Total Length',
                               'Max Euclidean Distance', 'Max Path Distance', 'Max Branch Order', ]
feature_name_maps = {
    'Number of Branches': 'No. of Branches',
    'Total Length': 'Length (μm)',
    'Max Path Distance': 'Max Path Dist. (μm)',
    'N_stem': 'No. of Stems',
    'Number of Tips': 'No. of Tips',
    'Max Branch Order': 'Max Branch Order',
    'N_node': 'No. of Nodes',
    'Number of Bifurcatons': 'No. of Bifurcations',
    'Overall Width': 'Width (μm)',
    'Overall Height': 'Height (μm)',
    'Overall Depth': 'Depth (μm)',
    'Max Euclidean Distance': 'Max Euclidean (μm)',
    # 'Max Branch Order': 'Max Branch Order',
    # 'Average Bifurcation Angle Remote': 'Avg. Remote BA (°)'
}

colors = ['#ff9999', '#66b3ff']

def plt_current_sample(sample_id=12370):
    save_root = "/data/kfchen/trace_ws/immunohistochemistry_test/test_samples"
    save_dir = os.path.join(save_root, f'{sample_id}')
    os.makedirs(save_dir, exist_ok=True)
    
    img_dir = "/data/kfchen/trace_ws/de_flu_test/14k_tif"
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    img_files = [f for f in img_files if int(f.split('_')[0]) == sample_id]
    img_file = img_files[0]
    shutil.copy(os.path.join(img_dir, img_file), os.path.join(save_dir, str(sample_id) + '_img.tif'))

    best_sigma_file = os.path.join(save_dir, str(sample_id) + '_best_sigma.tif')
    get_best_sigma_map(os.path.join(img_dir, img_file), best_sigma_file)

    enhanced_img_dir = "/data/kfchen/trace_ws/de_flu_test/de_tif_v2_14k"
    enhanced_img_files = [f for f in os.listdir(enhanced_img_dir) if f.endswith('.tif')]
    enhanced_img_files = [f for f in enhanced_img_files if int(f.split('_')[0]) == sample_id]
    enhanced_img_file = enhanced_img_files[0]
    shutil.copy(os.path.join(enhanced_img_dir, enhanced_img_file), os.path.join(save_dir, str(sample_id) + '_enhanced_img.tif'))

    seg_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/0_seg"
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.tif')]
    seg_files = [f for f in seg_files if int(f.split('_')[0]) == sample_id]
    seg_file = seg_files[0]
    shutil.copy(os.path.join(seg_dir, seg_file), os.path.join(save_dir, str(sample_id) + '_seg.tif'))

    skel_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/3_skel_with_soma"
    skel_files = [f for f in os.listdir(skel_dir) if f.endswith('.tif')]
    skel_files = [f for f in skel_files if int(f.split('_')[0]) == sample_id]
    skel_file = skel_files[0]
    shutil.copy(os.path.join(skel_dir, skel_file), os.path.join(save_dir, str(sample_id) + '_skel.tif'))

    swc_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc"
    swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
    swc_files = [f for f in swc_files if int(f.split('_')[0]) == sample_id]
    swc_file = swc_files[0]
    shutil.copy(os.path.join(swc_dir, swc_file), os.path.join(save_dir, str(sample_id) + '_swc.swc'))

    img = tifffile.imread(os.path.join(img_dir, img_file))
    enhanced_img = tifffile.imread(os.path.join(enhanced_img_dir, enhanced_img_file))
    seg = tifffile.imread(os.path.join(seg_dir, seg_file))
    skel = tifffile.imread(os.path.join(skel_dir, skel_file))

    mip_list = [
        np.max(img, axis=0),
        np.max(enhanced_img, axis=0),
        np.max(seg, axis=0),
        np.max(skel, axis=0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    for i, (mip, title) in enumerate(zip(mip_list, ['img', 'enhanced_img', 'seg', 'skel'])):
        ax = axes[i]
        ax.imshow(mip, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, str(sample_id) + '_mip.png'))
    plt.close()

def crop_or_pad_2d(img, target_shape):
    img_shape = img.shape
    pad_width = [(0, 0) for _ in range(len(img_shape))]
    for i in range(len(img_shape)):
        if img_shape[i] < target_shape[i]:
            pad_width[i] = int((target_shape[i] - img_shape[i]) / 2)
    img = np.pad(img, pad_width, mode='constant', constant_values=0)

    # crop
    img_shape = img.shape
    crop_start = [0 for _ in range(len(img_shape))]
    for i in range(len(img_shape)):
        if img_shape[i] > target_shape[i]:
            crop_start[i] = int((img_shape[i] - target_shape[i]) / 2)
    img = img[crop_start[0]:crop_start[0] + target_shape[0], crop_start[1]:crop_start[1] + target_shape[1]]

    return img


def plt_random_sample(no_immu_list, do_immu_list, meta_info, random_sample_size=5):
    img_dir = "/data/kfchen/trace_ws/de_flu_test/14k_tif"
    no_immu_list = random.sample(no_immu_list, random_sample_size)
    no_immu_list = [1364, 422, 1351, 1831, 2297] #
    do_immu_list = random.sample(do_immu_list, random_sample_size)
    do_immu_list = [7382, 7964, 8599, 7403, 6769]
    traced_neuron_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv"
    traced_neuron_list = pd.read_csv(traced_neuron_list_file)['id'].tolist()
    # 检查是否都在traced_neuron_list中
    print("if all in traced_neuron_list")
    print(set(no_immu_list) & set(traced_neuron_list))
    print(set(do_immu_list) & set(traced_neuron_list))

    # meta_info_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
    # meta_info = pd.read_excel(meta_info_file)
    # xy_resolution = meta_info[meta_info['cell_id'] == int(os.path.basename(img_file).split('_')[0])]['xy_resolution'].values[0]
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    img_files = [f for f in img_files if int(f.split('_')[0]) in no_immu_list + do_immu_list]


    target_img_size = (300, 300, 300)
    mip_list_wo, mip_list_w = [], []

    for id in no_immu_list + do_immu_list:
        img_file = [f for f in img_files if int(f.split('_')[0]) == id][0]
        img = tifffile.imread(os.path.join(img_dir, img_file))
        img = img.astype(np.float32)
        img = (img -  img.min()) / (img.max() - img.min()) * 255
        xy_resolution = meta_info[meta_info['cell_id'] == int(os.path.basename(img_file).split('_')[0])]['xy_resolution'].values[0]
        print(img.shape, xy_resolution)
        img = resize(img, (img.shape[0], int(img.shape[1] * xy_resolution / 1000.0), int(img.shape[2] * xy_resolution / 1000.0)), order=1)
        print(img.shape, '\n')
        img = (img - img.min()) / (img.max() - img.min())*255
        img = img.astype(np.uint8)
        img_mip = np.max(img, axis=0)
        # img_mip = crop_or_pad_2d(img_mip, (int(1000.0 * img_size[1] / xy_resolution), int(1000.0 * img_size[2] / xy_resolution)))
        print(img_mip.shape)
        img_mip = crop_or_pad_2d(img_mip, target_img_size[:2])
        print(img_mip.shape)
        if(int(img_file.split('_')[0]) in do_immu_list):
            mip_list_w.append((id, img_mip))
        else:
            mip_list_wo.append((id, img_mip))

    fig, axes = plt.subplots(2, random_sample_size, figsize=(random_sample_size * 3, 6), dpi=300)
    for i, mip in enumerate(mip_list_wo):
        id, mip = mip
        ax = axes[0, i]
        ax.imshow(mip, cmap='gray')
        ax.axis('off')
        ax.set_title(f'No. {id :05d}', fontsize=15, color='white', transform=ax.transAxes, x=0.5, y=0.9, ha='center', va='center')


    for i, mip in enumerate(mip_list_w):
        id, mip = mip
        ax = axes[1, i]
        ax.imshow(mip, cmap='gray')
        ax.axis('off')
        ax.set_title(f'No. {id :05d}', fontsize=15, color='white', transform=ax.transAxes, x=0.5, y=0.9, ha='center', va='center')

    plt.tight_layout()
    plt.savefig('/data/kfchen/trace_ws/immunohistochemistry_test/random_sample.png')
    plt.close()

def get_best_sigma_map(img_file, sigam_map_file):
    img = tifffile.imread(img_file)
    origin_img_shape = img.shape

    # meta_info_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
    # meta_info = pd.read_excel(meta_info_file)
    # xy_resolution = meta_info[meta_info['cell_id'] == int(os.path.basename(img_file).split('_')[0])]['xy_resolution'].values[0]
    xy_resolution = 919.0

    img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
    resolution = (1, xy_resolution / 1000, xy_resolution / 1000)
    print("resize img begin")
    img = resize(img, (img.shape[0] * resolution[0], img.shape[1] * resolution[1], img.shape[2] * resolution[2]),
                 order=1)
    print("resize img end")
    # img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)

    soma = np.where(img > 0.9, img, 0).astype(np.float32)

    print("find_best_sigma_map begin")
    best_sigma_map, best_sigma = find_best_sigma_map(img, soma)
    print("find_best_sigma_map end")
    print("resize best_sigma_map begin")
    best_sigma_map = resize(best_sigma_map, origin_img_shape, order=1, preserve_range=True,
                            anti_aliasing=False).astype(best_sigma_map.dtype)
    print("resize best_sigma_map end")
    best_sigma_map = (best_sigma_map - best_sigma_map.min()) / (best_sigma_map.max() - best_sigma_map.min()) * 255
    best_sigma_map = best_sigma_map.astype("uint8")
    tifffile.imwrite(sigam_map_file, best_sigma_map)

def plot_samples(sample_list, mip_dir = "/data/kfchen/trace_ws/de_flu_test/mip_v2_14k"):
    os.makedirs(mip_dir, exist_ok=True)

    img_dir = "/data/kfchen/trace_ws/de_flu_test/14k_tif"
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    img_files = [f for f in img_files if int(f.split('_')[0]) in sample_list]
    print(len(img_files))

    def current_task(img_file, mip_file):
        img = tifffile.imread(os.path.join(img_dir, img_file))
        img = (img - img.min()) / (img.max() - img.min())*255
        img = img.astype(np.uint8)
        mip = np.max(img, axis=0)
        tifffile.imwrite(mip_file, mip)

    Parallel(n_jobs=8)(delayed(current_task)(
        os.path.join(img_dir, img_file),
        os.path.join(mip_dir, img_file.replace('.tif', '.png'))
    ) for img_file in tqdm(img_files))



def plot1(no_immu_list, do_immu_list, save_file='/data/kfchen/trace_ws/immunohistochemistry_test/a.png'):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    ax.pie(
        [len(no_immu_list), len(do_immu_list)],  # 数量
        autopct='',
        wedgeprops={'width': 0.5, 'edgecolor': 'black', 'linewidth': 0.5, 'alpha': 0.5},
        startangle=0,  # 设置开始角度，便于查看
        # 颜色
        colors=colors,
    )
    legend_labels = []
    for i in range(2):
        percentage = float([len(no_immu_list), len(do_immu_list)][i]) / len(traced_neuron_list) * 100
        legend_labels.append(f'{["W/O IHC", "W/ IHC"][i]}; n={[len(no_immu_list), len(do_immu_list)][i]} ({percentage:.2f}%)')
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, ncol=1, frameon=False,
                  shadow=True)
    plt.subplots_adjust(left=-0.5)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(save_file)
    plt.close()

def plot2(save_file):
    col = 6
    row = 2
    fig, axes = plt.subplots(row, col, figsize=(col * 2.5, row * 3), dpi=300)
    axes = axes.flatten()
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        positions = range(2)
        violin_parts = ax.violinplot(
            [no_immu_l_measure_result[feature_name].values, do_immu_l_measure_result[feature_name].values],
            positions=positions, widths=0.6,
            showmeans=False, showmedians=False, showextrema=False,
            )
        for partname in ['bodies']:
            # for part in violin_parts[partname]:
            for i, part in enumerate(violin_parts[partname]):
                part.set_edgecolor('black')  # 设置边缘线的颜色
                part.set_linewidth(1)  # 设置边缘线的宽度
                part.set_facecolor(colors[i])  # 设置填充颜色
                part.set_alpha(0.5)
        # box
        if (feature_name == "Max Euclidean Distance" and "proposed_1um_l_measure_total" in l_measure_result_file):
            box_width = 0.1
        else:
            box_width = 0.2
        ax.boxplot([no_immu_l_measure_result[feature_name].values, do_immu_l_measure_result[feature_name].values],
                   positions=positions, widths=box_width,
                   patch_artist=True,
                   showfliers=True,
                   boxprops=dict(color='black', linewidth=1, facecolor='white'),
                   capprops=dict(color='black'),
                   medianprops=dict(color='black'),
                   flierprops=dict(marker='o', color='black', markersize=3)
                   )
        ax.set_xticks(positions)
        # ax.set_xticklabels([f'W/O IHC ({float(len(no_immu_list)) / (len(no_immu_list) + len(do_immu_list)):.2f})',
        #                     f'W/ IHC\n({len(do_immu_list) / (len(no_immu_list) + len(do_immu_list)):.2f})'])
        ax.set_xticklabels(['W/O IHC', 'W/ IHC'], rotation=0)
        # ax.set_title(feature_name_maps[feature_name])
        ax.set_ylabel(feature_name_maps[feature_name], fontsize=12)
        # print mean
        print(
            f'{feature_name}: W/O IHC mean={no_immu_l_measure_result[feature_name].mean()}, W/ IHC mean={do_immu_l_measure_result[feature_name].mean()}')
        increased_percentage = (
                    (do_immu_l_measure_result[feature_name].mean() - no_immu_l_measure_result[feature_name].mean())
                    / no_immu_l_measure_result[feature_name].mean() * 100)
        if(increased_percentage > 0):
            plot_text = f'+{increased_percentage:.2f}%'
        else:
            plot_text = f'{increased_percentage:.2f}%'
        ax.text(0.5, 0.9, plot_text, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red')
        # ylim + 10%
        y_min = min(no_immu_l_measure_result[feature_name].min(), do_immu_l_measure_result[feature_name].min())
        y_max = max(no_immu_l_measure_result[feature_name].max(), do_immu_l_measure_result[feature_name].max())
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    axes[-1].axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_file)
    plt.close()

def get_img_shape(id, img_dir, shape_save_dir="/data/kfchen/trace_ws/immunohistochemistry_test/img_shape"):
    # id = str(int(os.path.basename(id).split("_")[0].split(".")[0]))
    img_shape_save_file = os.path.join(shape_save_dir, id + ".npz")
    if os.path.exists(img_shape_save_file):
        return np.load(img_shape_save_file)['img_shape']

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    img_file = [f for f in img_files if int(f.split("_")[0]) == int(id)][0]
    img = tifffile.imread(os.path.join(img_dir, img_file))
    img_shape = img.shape
    np.savez(img_shape_save_file, img_shape=img_shape)

    return img_shape



def calc_z_size(do_immu_list, no_immu_list):
    Parallel(n_jobs=8)(delayed(get_img_shape)(
        str(id), "/data/kfchen/trace_ws/de_flu_test/14k_tif"
    ) for id in tqdm(do_immu_list + no_immu_list))

    wihc_z_list, woihc_z_list = [], []
    for id in do_immu_list:
        img_shape = get_img_shape(str(id), "/data/kfchen/trace_ws/de_flu_test/14k_tif")
        wihc_z_list.append(img_shape[0])
    for id in no_immu_list:
        img_shape = get_img_shape(str(id), "/data/kfchen/trace_ws/de_flu_test/14k_tif")
        woihc_z_list.append(img_shape[0])

    woihc_z_slice_info, wihc_z_slice_info = np.array(woihc_z_list), np.array(wihc_z_list)

    print(
        f'W/O IHC slice thickness mean={woihc_z_slice_info.mean()}, W/ IHC slice thickness mean={wihc_z_slice_info.mean()}')

    # plot violin
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    positions = range(2)
    violin_parts = ax.violinplot(
        [woihc_z_slice_info, wihc_z_slice_info],
        positions=positions, widths=0.6,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for partname in ['bodies']:
        for i, part in enumerate(violin_parts[partname]):
            part.set_edgecolor('black')
            part.set_linewidth(1)
            part.set_facecolor(colors[i])
            part.set_alpha(0.5)
    # box
    # ax.boxplot([woihc_z_slice_info, wihc_z_slice_info],
    #             positions=positions, widths=0.2,
    #             patch_artist=True,
    #             showfliers=True,
    #             boxprops=dict(color='black', linewidth=1, facecolor='white'),
    #             capprops=dict(color='black'),
    #             medianprops=dict(color='black'),
    #             flierprops=dict(marker='o', color='black', markersize=3)
    #             )
    ax.set_xticks(positions)
    ax.set_xticklabels(['W/O IHC', 'W/ IHC'], rotation=0)
    ax.set_ylabel('z size (μm)', fontsize=12)
    increased_percentage = (
            (wihc_z_slice_info.mean() - woihc_z_slice_info.mean())
            / woihc_z_slice_info.mean() * 100)
    if (increased_percentage > 0):
        plot_text = f'+{increased_percentage:.2f}%'
    else:
        plot_text = f'{increased_percentage:.2f}%'
    ax.text(0.5, 0.9, plot_text, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=12, color='red')
    y_min = min(woihc_z_slice_info.min(), wihc_z_slice_info.min())
    y_max = max(woihc_z_slice_info.max(), wihc_z_slice_info.max())
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    plt.tight_layout()
    plt.savefig('/data/kfchen/trace_ws/immunohistochemistry_test/z_shape.png')
    plt.close()

def get_250_list(do_immu_list, no_immu_list):
    neuron_meta_info_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
    neuron_meta_info_df = pd.read_excel(neuron_meta_info_file)
    z_slice_info = neuron_meta_info_df[['cell_id', 'slice_thickness']]
    wihc_z_slice_info = z_slice_info[z_slice_info['cell_id'].isin(do_immu_list)]['slice_thickness'].tolist()
    woihc_z_slice_info = z_slice_info[z_slice_info['cell_id'].isin(no_immu_list)]['slice_thickness'].tolist()

    wihc_z_slice_df = z_slice_info[z_slice_info['cell_id'].isin(do_immu_list)]
    woihc_z_slice_df = z_slice_info[z_slice_info['cell_id'].isin(no_immu_list)]
    # 检查各个值
    # print(wihc_z_slice_df.describe())
    # print(woihc_z_slice_df.describe())

    wihc_z_slice_df_250 = wihc_z_slice_df[wihc_z_slice_df['slice_thickness'] == 250]
    woihc_z_slice_df_250 = woihc_z_slice_df[woihc_z_slice_df['slice_thickness'] == 250]
    print(len(wihc_z_slice_df_250), len(woihc_z_slice_df_250))

    return wihc_z_slice_df_250['cell_id'].tolist(), woihc_z_slice_df_250['cell_id'].tolist()

def calc_z_slice_thickness(do_immu_list, no_immu_list):
    neuron_meta_info_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
    neuron_meta_info_df = pd.read_excel(neuron_meta_info_file)
    z_slice_info = neuron_meta_info_df[['cell_id', 'slice_thickness']]
    wihc_z_slice_info = z_slice_info[z_slice_info['cell_id'].isin(do_immu_list)]['slice_thickness'].tolist()
    woihc_z_slice_info = z_slice_info[z_slice_info['cell_id'].isin(no_immu_list)]['slice_thickness'].tolist()
    #
    wihc_z_slice_info = np.array(wihc_z_slice_info).astype(float)
    woihc_z_slice_info = np.array(woihc_z_slice_info).astype(float)

    print(
        f'W/O IHC slice thickness mean={woihc_z_slice_info.mean()}, W/ IHC slice thickness mean={wihc_z_slice_info.mean()}')

    # plot violin
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    positions = range(2)
    violin_parts = ax.violinplot(
        [woihc_z_slice_info, wihc_z_slice_info],
        positions=positions, widths=0.6,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for partname in ['bodies']:
        for i, part in enumerate(violin_parts[partname]):
            part.set_edgecolor('black')
            part.set_linewidth(1)
            part.set_facecolor(colors[i])
            part.set_alpha(0.5)
    # box
    # ax.boxplot([woihc_z_slice_info, wihc_z_slice_info],
    #             positions=positions, widths=0.2,
    #             patch_artist=True,
    #             showfliers=True,
    #             boxprops=dict(color='black', linewidth=1, facecolor='white'),
    #             capprops=dict(color='black'),
    #             medianprops=dict(color='black'),
    #             flierprops=dict(marker='o', color='black', markersize=3)
    #             )
    ax.set_xticks(positions)
    ax.set_xticklabels(['W/O IHC', 'W/ IHC'], rotation=0)
    ax.set_ylabel('z size (μm)', fontsize=12)
    increased_percentage = (
            (wihc_z_slice_info.mean() - woihc_z_slice_info.mean())
            / woihc_z_slice_info.mean() * 100)
    if (increased_percentage > 0):
        plot_text = f'+{increased_percentage:.2f}%'
    else:
        plot_text = f'{increased_percentage:.2f}%'
    ax.text(0.5, 0.9, plot_text, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=12, color='red')
    y_min = min(woihc_z_slice_info.min(), wihc_z_slice_info.min())
    y_max = max(woihc_z_slice_info.max(), wihc_z_slice_info.max())
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    plt.tight_layout()
    plt.savefig('/data/kfchen/trace_ws/immunohistochemistry_test/slice_thickness.png')
    plt.close()


if __name__ == '__main__':
    # plt_current_sample()
    # exit()

    temp_save_file = '/data/kfchen/trace_ws/immunohistochemistry_test/immunohistochemistry_list.npz'
    if(os.path.exists(temp_save_file)):
        data = np.load(temp_save_file)
        do_immu_list = list(data['do_immu_list'])
        no_immu_list = list(data['no_immu_list'])
    else:
        print("prepare data")
        neuron_meta_info_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
        # neuron_meta_info_df = pd.read_excel(neuron_meta_info_file)
        neuron_meta_info_df = pd.read_excel(neuron_meta_info_file)
        immunohistochemistry_info = neuron_meta_info_df[['cell_id', 'immunohistochemistry']]
        do_immu_list = immunohistochemistry_info[immunohistochemistry_info['immunohistochemistry'] == '1'][
            'cell_id'].tolist()
        no_immu_list = immunohistochemistry_info[immunohistochemistry_info['immunohistochemistry'] == '0'][
            'cell_id'].tolist()
        others_list = immunohistochemistry_info[immunohistochemistry_info['immunohistochemistry'] == '--'][
            'cell_id'].tolist()

        traced_neuron_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv"
        traced_neuron_list = pd.read_csv(traced_neuron_list_file)['id'].tolist()
        # print(len(traced_neuron_list))

        do_immu_list = list(set(do_immu_list) & set(traced_neuron_list))
        no_immu_list = list(set(no_immu_list) & set(traced_neuron_list))
        np.savez(temp_save_file, do_immu_list=do_immu_list, no_immu_list=no_immu_list)


    # calc_z_slice_thickness(do_immu_list, no_immu_list)
    # calc_z_size(do_immu_list, no_immu_list)

    plt_random_sample(no_immu_list, do_immu_list, neuron_meta_info_df)
    plot1(no_immu_list, do_immu_list)

    print(f"full w/o IHC: {len(no_immu_list)}, full w/ IHC: {len(do_immu_list)}")
    do_immu_list, no_immu_list = get_250_list(do_immu_list, no_immu_list)
    print(f"thikness=250 w/o IHC: {len(no_immu_list)}, 250 w/ IHC: {len(do_immu_list)}")

    l_measure_result_file = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc_l_measure.csv"  # origin
    l_measure_result_df = pd.read_csv(l_measure_result_file)
    do_immu_l_measure_result = l_measure_result_df[l_measure_result_df['ID'].isin(do_immu_list)]
    no_immu_l_measure_result = l_measure_result_df[l_measure_result_df['ID'].isin(no_immu_list)]
    plot2('/data/kfchen/trace_ws/immunohistochemistry_test/immunohistochemistry.png')

    l_measure_result_file = "/data/kfchen/trace_ws/cropped_swc/proposed_1um_l_measure_total.csv" # cropped
    l_measure_result_df = pd.read_csv(l_measure_result_file)
    do_immu_l_measure_result = l_measure_result_df[l_measure_result_df['ID'].isin(do_immu_list)]
    no_immu_l_measure_result = l_measure_result_df[l_measure_result_df['ID'].isin(no_immu_list)]
    plot2('/data/kfchen/trace_ws/immunohistochemistry_test/immunohistochemistry_cropped.png')




