import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_ver_1():
    # 创建饼图，不显示百分比和标签
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.pie(sizes, labels=['' for i in range(len(sizes))], colors=colors, autopct='',
           wedgeprops={'width': 0.5, 'edgecolor': 'black', 'linewidth': 0.5}, )
    # sns绘制pie

    # 计算每个扇区的百分比并在 legend 中显示
    legend_labels = []
    for i in range(len(sizes)):
        percentage = sizes[i] / sum(sizes) * 100
        # 创建 legend 标签: 名称、数量、百分比
        legend_labels.append(f'{labels_abbr[i]}; n={sizes[i]} ({percentage:.2f}%)')

    # 将 legend 放置在图的右侧
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, ncol=2, frameon=False,
              shadow=True)

    plt.subplots_adjust(left=-0.5)

    # 显示图表
    plt.show()

def plot_ver_2(sizes, labels_abbr, colors):
    sorted_sizes, sorted_labels, sorted_colors = zip(*sorted(zip(sizes, labels_abbr, colors), reverse=True))
    sizes, labels_abbr, colors = list(sorted_sizes), list(sorted_labels), list(sorted_colors)
    # 创建饼图，不显示百分比和标签
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.pie(sizes, labels=['' for i in range(len(sizes))], colors=colors, autopct='',
           wedgeprops={'width': 0.5, 'edgecolor': 'black', 'linewidth': 0.5}, )
    # sns绘制pie

    # 计算每个扇区的百分比并在 legend 中显示
    legend_labels = []
    for i in range(len(sizes)):
        percentage = sizes[i] / sum(sizes) * 100
        # 创建 legend 标签: 名称、数量、百分比
        legend_labels.append(f'{labels_abbr[i]}; n={sizes[i]} ({percentage:.2f}%)')

    # 将 legend 放置在图的右侧
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, ncol=2, frameon=False,
              shadow=True)

    plt.subplots_adjust(left=-0.5)

    # 显示图表
    plt.show()



neuron_info_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
df = pd.read_csv(neuron_info_file)
brain_regions = df['brain_region'].tolist()
# print(brain_regions)
# exit()

brain_region_map = {
    'frontal lobe': ['FL.L', 'FL.R', 'FL_TL.L', 'FP.R', 'FP.L', ],
    "superior frontal gyrus": ["SFG.R", "SFG.L", "SFG", "S(M)FG.R", "M(I)FG.L"],
    "middle frontal gyrus": ["MFG.R", "MFG", "MFG.L"],
    "inferior frontal gyrus": ["IFG.R", "(X)FG", "IFG"],
    #
    'temporal lobe': ['TL.L', 'TL.R', "TP", "TP.L", "TP.R", 'FT.L'],
    "superior temporal gyrus": ["STG", "STG.R", "S(M)TG.R", "S(M)TG.L", "STG-AP", "S(M,I)TG"],
    "middle temporal gyrus": ["MTG.R", "MTG.L", "MTG"],

    "parietal lobe": ["PL.L", "PL.L_OL.L", "PL"],
    "inferior parietal lobe": ["IPL-near-AG", "IPL.L"],

    'occipital lobe': ['OL.L', 'OL.R'],


    # 'posterior lateral ventricle': ['pLV.L'],
    'others': ['CB_tonsil.L', 'BN.L', 'CC.L',],
}

mapped_brain_regions = []
for region in brain_regions:
    for key, value in brain_region_map.items():
        if region in value:
            mapped_brain_regions.append(key)
            break
print(f"len(mapped_brain_regions): {len(mapped_brain_regions)}, len(brain_regions): {len(brain_regions)}")


# 统计各个脑区的数量
labels = ["superior frontal gyrus", "middle frontal gyrus", "inferior frontal gyrus",
            "parietal lobe", "inferior parietal lobe",
            "superior temporal gyrus", "middle temporal gyrus",
            'occipital lobe',
            'temporal lobe',
            'frontal lobe',
            'others'
            ]
sizes = [mapped_brain_regions.count(label) for label in labels]

# 缩写
labels_abbr = ["SFG", "MFG", "IFG", "PL", "IPL", "STG", "MTG", "OL", "TL", "FL", "Others"]
# colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
colors = plt.cm.get_cmap('Set3').colors[:len(sizes)]

# plot_ver_1()
plot_ver_2(sizes, labels_abbr, colors)

