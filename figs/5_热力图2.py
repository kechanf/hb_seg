import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


train_val_list_file = "/data/kfchen/trace_ws/paper_trace_result/train_val_list.csv"
train_set_ids = pd.read_csv(train_val_list_file)['id'].tolist()

manual_l_measure_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv"
total_l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv"
manual_l_measure_df = pd.read_csv(manual_l_measure_file)
total_l_measure_df = pd.read_csv(total_l_measure_file)

recon_meta_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv"
recon_ids = pd.read_csv(recon_meta_file)['ID'].tolist()
unlabeled_ids = list(set(recon_ids) - set(train_set_ids))

# 比较manual：train_set_ids
# 以及auto：unlabeled_ids

neuron_meta_14k_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/Human_SingleCell_TrackingTable_20240712.csv"
neuron_meta_14k = pd.read_csv(neuron_meta_14k_file, encoding='gbk')

l_measure_df = pd.merge(manual_l_measure_df, total_l_measure_df, on='ID', how='outer')
neuron_meta_14k = pd.merge(neuron_meta_14k, l_measure_df, left_on='Cell ID', right_on='ID', how='left')


manual_meta_df = neuron_meta_14k[neuron_meta_14k['Cell ID'].isin(train_set_ids)]
unlabeled_meta_df = neuron_meta_14k[neuron_meta_14k['Cell ID'].isin(unlabeled_ids)]
print(f"manual_meta_df: {manual_meta_df.shape}")
print(f"unlabeled_meta_df: {unlabeled_meta_df.shape}")

manual_brain_regions = manual_meta_df['脑区'].unique()
unlabeled_brain_regions = unlabeled_meta_df['脑区'].unique()

print(f"manual_brain_regions: {manual_brain_regions}")
print(f"auto_brain_regions: {unlabeled_brain_regions}")
print("common brain regions:", set(manual_brain_regions) & set(unlabeled_brain_regions))
common_brain_regions = list(set(manual_brain_regions) & set(unlabeled_brain_regions))

filter_manual_meta_df = manual_meta_df[manual_meta_df['脑区'].isin(common_brain_regions)]
filter_unlabeled_meta_df = unlabeled_meta_df[unlabeled_meta_df['脑区'].isin(common_brain_regions)]
# 每个脑区各有多少个样本
manual_brain_region_counts = filter_manual_meta_df['脑区'].value_counts()
unlabeled_brain_region_counts = filter_unlabeled_meta_df['脑区'].value_counts()
print(manual_brain_region_counts)
print(unlabeled_brain_region_counts)

# 过滤出manual和unlabeled中样本数都大于10的脑区
filtered_regions = [
    region for region in common_brain_regions
    if manual_brain_region_counts.get(region, 0) > 10 and unlabeled_brain_region_counts.get(region, 0) > 10
]

print(f"Filtered brain regions (both manual and auto samples > 10): {filtered_regions}")

# 再次根据过滤后的脑区集合进行数据过滤
filter_manual_meta_df = filter_manual_meta_df[filter_manual_meta_df['脑区'].isin(filtered_regions)]
filter_unlabeled_meta_df = filter_unlabeled_meta_df[filter_unlabeled_meta_df['脑区'].isin(filtered_regions)]

# 打印过滤后的数据
print(f"Filtered manual_meta_df: {filter_manual_meta_df.shape}")
print(f"Filtered unlabeled_meta_df: {filter_unlabeled_meta_df.shape}")

filter_manual_meta_df = pd.merge(filter_manual_meta_df, manual_l_measure_df, left_on="Cell ID", right_on="ID", how="inner")
filter_unlabeled_meta_df = pd.merge(filter_unlabeled_meta_df, total_l_measure_df, left_on="Cell ID", right_on="ID", how="inner")

manual_avg_features = filter_manual_meta_df[filter_manual_meta_df['脑区'].isin(filtered_regions)].groupby('脑区').mean(numeric_only=True)
print("Manual average features for each brain region:")
print(manual_avg_features)

# 对 auto df 计算每个脑区的平均值
unlabeled_avg_features = filter_unlabeled_meta_df[filter_unlabeled_meta_df['脑区'].isin(filtered_regions)].groupby('脑区').mean(numeric_only=True)
print("Auto average features for each brain region:")
print(unlabeled_avg_features)

# 输出所有的列名
print("Manual average features columns:", manual_avg_features.columns.tolist())

meaningful_feasures = ['N_stem', 'Number of Bifurcatons', 'Number of Branches', 'Number of Tips',
                       'Overall Width', 'Overall Height', 'Overall Depth', 'Total Length',
                       'Max Euclidean Distance', 'Max Path Distance', 'Max Branch Order',
                       ]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

manual_avg_features = manual_avg_features[meaningful_feasures]
unlabeled_avg_features = unlabeled_avg_features[meaningful_feasures]
print(manual_avg_features)
print(unlabeled_avg_features)
# 对每一列进行归一化处理
manual_avg_features = (manual_avg_features - manual_avg_features.min()) / (manual_avg_features.max() - manual_avg_features.min())
unlabeled_avg_features = (unlabeled_avg_features - unlabeled_avg_features.min()) / (unlabeled_avg_features.max() - unlabeled_avg_features.min())

# 转置
manual_avg_features = manual_avg_features.T
unlabeled_avg_features = unlabeled_avg_features.T


# corr
manual_avg_features = manual_avg_features.corr()
unlabeled_avg_features = unlabeled_avg_features.corr()
# plot


print(manual_avg_features)
print(unlabeled_avg_features)

def plot_heatmap(data, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 完整显示


# Plot heatmap for manual average features
plot_heatmap(manual_avg_features, "Manual Average Features Correlation", "/home/kfchen/neuron_seg_human/fig_results/fig6c_heatmap_manual.png")
# Plot heatmap for unlabeled average features.
plot_heatmap(unlabeled_avg_features, "Unlabeled Average Features Correlation", "/home/kfchen/neuron_seg_human/fig_results/fig6c_heatmap_unlabeled.png")

# 提取上三角的值（不包括对角线）
corr_matrix1, corr_matrix2 = manual_avg_features.values, unlabeled_avg_features.values

mask = np.triu_indices_from(corr_matrix1, k=1)  # k=1 跳过对角线
x_values = corr_matrix1[mask]
y_values = corr_matrix2[mask]

# 创建散点图
plt.figure(figsize=(4,4))
sns.scatterplot(x=x_values, y=y_values, alpha=0.7, s=100)

# 计算回归线 + 回归区间（预测区间）
sns.regplot(x=x_values, y=y_values,
            scatter=False,
            ci=None,  # 禁用置信区间
            line_kws={'color': 'red', 'lw': 2, 'label': 'Regression Line'})

# plt.xlim(-0.2, 1.2)
# plt.ylim(-0.2, 1.2)

# 计算Pearson相关系数和p值
r, p = stats.pearsonr(x_values, y_values)
print(f"Pearson correlation coefficient: {r:.2f}, p-value: {p:.2e}")

plt.xlabel('Manual', fontsize=16)
plt.ylabel('Auto', fontsize=16)

plt.tight_layout()
plt.savefig('/home/kfchen/neuron_seg_human/fig_results/fig6c_scatter_plot.png')


