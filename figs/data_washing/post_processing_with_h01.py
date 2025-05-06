import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np

my_data_l_measure_result_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv"
h01_lmeasure_result_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/h01_l_measure.csv"

my_data_l_measure_result = pd.read_csv(my_data_l_measure_result_file)
h01_lmeasure_result = pd.read_csv(h01_lmeasure_result_file)

# print cols
print(my_data_l_measure_result.columns)
print(h01_lmeasure_result.columns)

# rename
h01_lmeasure_result.rename(columns={"Stems": "N_stem",
                                    "AverageContraction": "Average Contraction",
                                    "AverageParent-daughterRatio": "Average Parent-daughter Ratio",
                                    "AverageBifurcationAngleLocal": "Average Bifurcation Angle Local",
                                    "AverageBifurcationAngleRemote": "Average Bifurcation Angle Remote",
                                    "HausdorffDimension": "Hausdorff Dimension",
                                    "Length": "Total Length",
                                    "Branches": "Number of Branches"
                                    }, inplace=True)

interested_cols = [
    "N_stem",
    # "Average Contraction",
    "Average Parent-daughter Ratio",
    # "Average Bifurcation Angle Local",
    # "Average Bifurcation Angle Remote",
    # "Hausdorff Dimension",
    # "Total Length",
    # "Number of Branches"
]

# filter interested cols
my_data_l_measure_result = my_data_l_measure_result[interested_cols]
h01_lmeasure_result = h01_lmeasure_result[interested_cols]
deleted_counts = {}
thresholds = {}

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
ax = ax.flatten()

for col in interested_cols:
    idx = interested_cols.index(col)

    # 计算可信分布h01的5%左侧临界值
    threshold = np.percentile(h01_lmeasure_result[col], 5)  # 5%分位数
    thresholds[col] = threshold


    # 在my_data中标记小于threshold的样本
    original_sample_count = len(my_data_l_measure_result[col])
    filtered_data = my_data_l_measure_result[col][my_data_l_measure_result[col] >= threshold]
    deleted_sample_count = original_sample_count - len(filtered_data)

    # 保存删除数量
    deleted_counts[col] = deleted_sample_count

    # 画图
    ax[idx].hist(my_data_l_measure_result[col], bins=20, alpha=0.5, label="My Data (Before Filter)", density=True)
    ax[idx].hist(h01_lmeasure_result[col], bins=20, alpha=0.5, label="H01 Data (Reference)", density=True)

    # 画出滤波线
    ax[idx].axvline(threshold, color='red', linestyle='--', label=f'5% Threshold ({threshold:.2f})')

    # 设置标题和标签
    ax[idx].set_title(col)
    ax[idx].set_xlabel(col)
    ax[idx].set_ylabel("Frequency")

    ax[idx].text(0.5, 0.5, f"Deleted: {deleted_sample_count}", transform=ax[idx].transAxes,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # 设置图例
    ax[idx].legend()
# set title

plt.suptitle("L-measure result comparison between my data and H01 data")


plt.tight_layout()
plt.savefig("/home/kfchen/neuron_seg_human/fig_results/post_processing_with_h01.png")


# 开始做滤波
total_l_measure = pd.read_csv(my_data_l_measure_result_file)

# 根据自己的5%阈值进行滤波
self_filter_cols = [
    "Total Length",
    "Number of Branches",
]

h01_filter_cols = [
    "N_stem",
    # "Average Contraction",
    "Average Parent-daughter Ratio",
    # "Average Bifurcation Angle Local",
    # "Average Bifurcation Angle Remote",
    # "Hausdorff Dimension",
]
print(thresholds)
for col in self_filter_cols:
    threshold = np.percentile(total_l_measure[col], 5)  # 5%分位数
    thresholds[col] = threshold

for col in h01_filter_cols+self_filter_cols:
    if(not col in thresholds):
        continue
    threshold = thresholds[col]  # 5%分位数
    total_l_measure = total_l_measure[total_l_measure[col] >= threshold]
    print(f"Filtering {col} with threshold {threshold:.2f}")

print(f"Total samples after filtering: {len(total_l_measure)}")

