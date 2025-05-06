import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from collections import Counter

# 完整显示
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# csv_files = ["train_val_list.csv", "test_list_with_gs.csv", "unlabeled_list.csv"]
# total_neuron_id_list = []
# for csv_file in csv_files:
#     df = pd.read_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", csv_file))
#     current_id_list = df['id'].tolist()
#     total_neuron_id_list.extend(current_id_list)
# print("total_neuron_id_list: ", len(total_neuron_id_list))
neuron_meta_14k_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
neuron_meta_14k = pd.read_csv(neuron_meta_14k_file, encoding='gbk')
print(neuron_meta_14k.shape)
recon_ids = neuron_meta_14k['cell_id'].tolist()
# 检查是否有重复的id
# print(len(recon_ids), len(set(recon_ids)))
# # 找到重复的id
#
# id_counter = Counter(recon_ids)
# duplicate_ids = [k for k, v in id_counter.items() if v > 1]
# print(duplicate_ids)

# final_recon_list = neuron_meta_14k[neuron_meta_14k['Cell ID'].isin(total_neuron_id_list)]
final_recon_list = neuron_meta_14k
print(final_recon_list.shape)
recon_brain_regions = final_recon_list['brain_region'].unique()
# print(recon_brain_regions)

brain_regions = {
    'Frontal': {
        "Frontal body": ["SFG.R", "SFG.L", "SFG"],  # 和superior frontal gyrus是同一个区域
        "Middle frontal": ["MFG.R", "MFG", "MFG.L"],
        "Inferior frontal": ["IFG", "IFG.R"],
        "Frontal pole": ['FP.L', 'FP.R'],
        "Ambiguous (Frontal lobe)": ['FL.L', 'FL.R', '(X)FG', 'M(I)FG.L', 'S(M)FG.R', ],  # 后面两个是交叉脑区

        # "frontal tubercle": ['FT.L'], # 不在allen的atlas里面
    },
    'Parietal': {
        # "superior parietal gyrus": ["SPG.R", "SPG.L", "SPG"], # 9
        # 'inferior parietal gyrus': ["IPL", "IPL.L", 'IPL-near-AG'], # 10
        "Parietal": ["PL.L", "PL"],
        'Supramarginal': ["IPL", "IPL.L", 'IPL-near-AG'],  # 指的应该是同一个区域
         # # 16 13 31 51
    },
    'Temporal': {
        "Temporal body": ["STG.R", "STG", 'STG-AP', "S(M)TG.R", 'S(M)TG.L', "MTG.R", "MTG.L", "MTG"],
        # superior temporal gyrus
        # "middle temporal gyrus": ["MTG.R", "MTG.L", "MTG"], # 28
        # "inferior temporal gyrus": [], # 3
        "Temporal pole": ["TP.R", "TP", "TP.L"],
        "Ambiguous (Temporal lobe)": ['TL.L', 'TL.R', 'S(M,I)TG', ]  # 后面三个是交叉脑区
    },
    'Occipital': {
        "Occipital": ['OL.L', 'OL.R']
    },
    "Ambiguous": {
        "Ambiguous": ["PL.L_OL.L", "FL_TL.L"]  # 不在allen的atlas里面
    }
}

CerebrA_atlas = {
    "superior frontal gyrus": 38,
    "middle frontal gyrus":[42, 1],
    "inferior frontal gyrus":0,
    "frontal pole": 0,

    "frontal tubercle":0,

    "inferior parietal gyrus": 0,

    "superior temporal gyrus": 45,
    "middle temporal gyrus":28,
    "temporal pole":0,
}

# unknown brain regions: {'S(M)TG.L', , 'S(M)TG.R',
# 'M(I)FG.L', 'FT.L', 'STG-AP', 'S(M,I)TG',
# 'S(M)FG.R', 'IPL-near-AG', , 'PL.L_OL.L'}
brain_region_mapping = {
    'frontal lobe': 'FL',
    'parietal lobe': 'PL',
    'temporal lobe': 'TL',
    'occipital lobe': 'OL',
}

def prepare_data():
    print(f"total recon samples: {final_recon_list.shape[0]}")

    total_brain_regions = []
    for k, v in brain_regions.items():
        for k1, v1 in v.items():
            total_brain_regions.extend(v1)
    print(total_brain_regions)

    print(f"known brain regions: {set(recon_brain_regions) & set(total_brain_regions)}")
    # print(f"unknown brain regions: {set(recon_brain_regions) - set(total_brain_regions)}")
    unknown_brain_regions = list(set(recon_brain_regions) - set(total_brain_regions))
    print(f"unknown brain regions: {unknown_brain_regions}")

    patient_info_file = "/data/kfchen/trace_ws/patient_info.xlsx"
    patient_info = pd.read_excel(patient_info_file)
    unknown_brain_regions_info = patient_info[patient_info['english_abbr_nj'].isin(unknown_brain_regions)][["english_abbr_nj", "english_full_name", "intracranial_location"]]
    unique_unknown_brain_regions_info = unknown_brain_regions_info.drop_duplicates()

    kown_brain_regions = list(set(recon_brain_regions) & set(total_brain_regions))
    final_recon_list_in_known_brain_regions = final_recon_list[final_recon_list['brain_region'].isin(kown_brain_regions)]
    print(final_recon_list_in_known_brain_regions['brain_region'].value_counts())
    num_0_brain_regions = set(total_brain_regions) - set(final_recon_list_in_known_brain_regions['brain_region'].unique())
    print(f"num_0_brain_regions: {num_0_brain_regions}")


    print(unique_unknown_brain_regions_info)

    final_recon_list_in_unknown_brain_regions = final_recon_list[final_recon_list['brain_region'].isin(unknown_brain_regions)]
    # discribe the number of neurons in each brain region
    print(final_recon_list_in_unknown_brain_regions['brain_region'].value_counts())

def get_certain_brain_region(region_name_list, mask_color_list, atlas_dict, itk_label, data, alpha=0.5):
    mip = np.max(data[98:, :, :], axis=0)
    mip = mip.astype(np.float32)
    mip = (mip - np.min(mip)) / (np.max(mip) - np.min(mip))
    mip = (1 - mip * 0.25)
    colored_mip = np.stack([mip, mip, mip], axis=-1)

    # data = data[:98, :, :]
    print("region_name_list: ", region_name_list)
    for region_name, mask_color in zip(region_name_list, mask_color_list):
        target_code_list = atlas_dict[atlas_dict['Region'] == region_name]['Code'].values
        target_voxel_value = itk_label[itk_label['LABEL'].isin(target_code_list)]['IDX'].values

        target_mask = np.isin(data, target_voxel_value)
        print("target_voxel_value, target_mask: ", target_voxel_value, np.sum(target_mask))
        mask_mip = np.max(target_mask, axis=0)
        # print(colored_mip.shape, mask_mip.shape)
        for axis in range(3):
            colored_mip[mask_mip, axis] = mask_color[axis] * alpha + colored_mip[mask_mip, axis] * (1 - alpha)

    colored_mip = colored_mip[20:-20, 20:-20, :]
    # colored_mip = cv2.cvtColor(colored_mip, cv2.COLOR_RGB2BGR)
    colored_mip = np.transpose(colored_mip, (0, 1, 2))
    colored_mip = np.rot90(colored_mip)
    return colored_mip

    # plt.figure(figsize=(10, 10), dpi=300)
    # plt.axis('off')
    # plt.imshow(colored_mip)
    # plt.tight_layout()
    #
    # plt.show()
    # plt.close()

def show_yale_atlas():
    mask_mgz_file = "/data/kfchen/trace_ws/atlas/yale/YBA_696.nii"
    atlas_dict_file = "/data/kfchen/trace_ws/atlas/yale/Atlas_Dict.json"
    atlas_dict = pd.read_json(atlas_dict_file)

    itk_label_file = "/data/kfchen/trace_ws/atlas/yale/YBA_696_ITKlabels.txt"
    # IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
    itk_label = pd.read_csv(
        itk_label_file,
        comment="#",  # 跳过注释行
        sep="\s+",    # 使用正则表达式匹配任意空白字符
        header=None,  # 文件没有表头
        names=["IDX", "R", "G", "B", "A", "VIS", "MSH", "LABEL"]  # 指定列名
    )

    img = nib.load(mask_mgz_file)
    data = img.get_fdata()
    # plt.imshow(np.max(data, axis=0))
    # plt.show()

    set2_colors = plt.cm.get_cmap('tab10').colors
    print(len(set2_colors))
    # remove the 7th color
    set2_colors = np.concatenate((set2_colors[:7], set2_colors[8:]), axis=0)
    col, row = 1, 5
    fig, axs = plt.subplots(col, row, figsize=(row * 5, col * 5), dpi=300)
    axs = axs.flatten()
    mask_color_list = set2_colors


    for i, lobe in enumerate(brain_regions.keys()):
        region_names = []
        for region_name in brain_regions[lobe].keys():
            if("ambiguous" not in region_name):
                region_names.append(region_name)
        print(region_names)
        # region_names = list(brain_regions["temporal"].keys())
        # if("ambiguous" in region_names):
        #     region_names.remove("ambiguous")

        colored_mip = get_certain_brain_region(region_names, mask_color_list, atlas_dict, itk_label, data)
        axs[i].imshow(colored_mip)
        # axs[0].set_title(brain_region_full_name)
        axs[i].axis('off')

        mask_color_list = mask_color_list[len(region_names):]
    # for i, brain_region_full_name in enumerate(brain_region_full_names):
    #     mask_color = set2_colors(i)[:3]
    #     colored_mip = get_certain_brain_region(brain_region_full_name, mask_color, atlas_dict, itk_label, data)
    #     if(colored_mip is not None):
    #         axs[i].imshow(colored_mip)
    #         axs[i].set_title(brain_region_full_name)
    #         axs[i].axis('off')
    #     else:
    #         print(f"{brain_region_full_name} not found")

    plt.tight_layout()
    plt.savefig("/data/kfchen/trace_ws/atlas/yale/yale_atlas.png")
    plt.close()


# prepare_data()

show_yale_atlas()
# 统计
commont_map = {}
def count():
    print(final_recon_list.shape)
    lobe_count = {}
    region_count = {}
    for lobe in brain_regions.keys():
        current_lobe_count = 0
        for region_name in brain_regions[lobe].keys():
            current_region_count = final_recon_list[final_recon_list['brain_region'].isin(brain_regions[lobe][region_name])].shape[0]
            current_lobe_count += current_region_count
            region_count[region_name] = current_region_count
        lobe_count[lobe] = current_lobe_count

    print(lobe_count)
    print(region_count)
    # merge
    export_csv_file =  "/data/kfchen/trace_ws/atlas/yale/region_count.csv"
    region_count_df = pd.DataFrame(region_count.items(), columns=['region_name', 'count'])
    region_count_df['percent'] = region_count_df['count'] / sum(region_count_df['count'])
    # 调整格式为2位百分数
    region_count_df['percent'] = region_count_df['percent'].apply(lambda x: format(x, '.1%'))
    # f"{region_name}, n={count} ({percent})"
    region_count_df['comment'] = region_count_df.apply(lambda x: f"{x['region_name']}, n={x['count']} ({x['percent']})", axis=1)
    commont_map = region_count_df['comment'].to_dict()
    lobe_count_df = pd.DataFrame(lobe_count.items(), columns=['lobe_name', 'count'])
    lobe_count_df['percent'] = lobe_count_df['count'] / sum(lobe_count_df['count'])
    lobe_count_df['percent'] = lobe_count_df['percent'].apply(lambda x: format(x, '.1%'))
    lobe_count_df['comment'] = lobe_count_df.apply(lambda x: f"{x['lobe_name']}, n={x['count']} ({x['percent']})", axis=1)
    commont_map = {**commont_map, **lobe_count_df['comment'].to_dict()}

    region_count_df.to_csv(export_csv_file, index=False)
    lobe_count_df.to_csv(export_csv_file, mode='a', index=False)
count()

label_lobe_map, label_region_map = {}, {}
for lobe in brain_regions.keys():
    for region_name in brain_regions[lobe].keys():
        for label in brain_regions[lobe][region_name]:
            label_lobe_map[label] = lobe
            label_region_map[label] = region_name
final_recon_list['lobe'] = final_recon_list['brain_region'].map(label_lobe_map)
final_recon_list['region'] = final_recon_list['brain_region'].map(label_region_map)

# 画图
def plot_brain_region():
    # print(f"male: {len(male_list)}, ratio: {len(male_list) / len(df) * 100:.2f}%")
    # print(f"male: {len(female_list)}, ratio: {len(female_list) / len(df) * 100:.2f}%")

    # age_gender_distribution = df.groupby(['AgeGroup', 'gender']).size().unstack(fill_value=0)
    lobe_region_distribution = final_recon_list.groupby(['lobe', 'region']).size().unstack(fill_value=0)
    # 特定顺序
    columns_in_order = []
    for lobe in brain_regions.keys():
        for region_name in brain_regions[lobe].keys():
            columns_in_order.append(region_name)
    lobe_region_distribution = lobe_region_distribution.reindex(index=brain_regions.keys(), columns=columns_in_order)



    # 设定颜色代码
    set2_colors = plt.cm.get_cmap('tab10').colors
    colors = np.concatenate((
        set2_colors[:4], [set2_colors[7]],
        set2_colors[4:6],
        [set2_colors[6]], [set2_colors[8]], [set2_colors[7]],
        [set2_colors[9]],
        [set2_colors[7]]
    ), axis=0)
    plt.rcParams['savefig.dpi'] = 300

    # 绘制堆叠柱状图
    ax = lobe_region_distribution.plot(kind='bar', stacked=True, figsize=(6 + 1, 4 + 1),
                            color=colors,
                            align='edge',
                            alpha=0.5,
                            edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Lobe', fontsize=15)
    ax.set_ylabel('Number of Neurons', fontsize=15)
    plt.xticks(rotation=45, ha='center', x=0.05)

    ax.legend(frameon=False, fontsize=15,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.5),
              # ncol=2,
              # loc='right',
              ncol=1,
              )
    # set legend no visible
    ax.get_legend().remove()
    plt.tick_params(axis='both', which='major', labelsize=15)  # 调整刻度标签大小
    plt.tight_layout()
    for container in ax.containers:
        plt.setp(container, width=0.5)  # 设置条形图中条的宽度

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.subplots_adjust(bottom=0.3)
    plt.savefig("/data/kfchen/trace_ws/atlas/yale/lobe_region_distribution.png")
    plt.close()

plot_brain_region()




