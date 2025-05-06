import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


df_a = pd.read_csv(r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc_l_measure.csv")
df_b = pd.read_csv(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv")

ids1 = df_a['ID'].tolist()
ids2 = df_b['ID'].tolist()

common_ids = list(set(ids1) & set(ids2))
df_a = df_a[df_a['ID'].isin(common_ids)]
df_b = df_b[df_b['ID'].isin(common_ids)]
# sort
df_a = df_a.sort_values(by='ID')
df_b = df_b.sort_values(by='ID')

df_a = df_a.reset_index(drop=True)
df_b = df_b.reset_index(drop=True)

meaningful_feasures = ['N_stem', 'Number of Bifurcatons', 'Number of Branches', 'Number of Tips',
                       'Overall Width', 'Overall Height', 'Overall Depth', 'Total Length',
                       'Max Euclidean Distance', 'Max Path Distance', 'Max Branch Order',
                       ]
df_a = df_a[meaningful_feasures]
df_b = df_b[meaningful_feasures]
# 归一化
# 对每一列做归一化
df_a = (df_a - df_a.min()) / (df_a.max() - df_a.min())
df_b = (df_b - df_b.min()) / (df_b.max() - df_b.min())

df_a = df_a.T
df_b = df_b.T
print(df_a.shape, df_b.shape)

df_a_corr = df_a.corr()
df_b_corr = df_b.corr()

manual_avg_features = df_a_corr
unlabeled_avg_features = df_b_corr

# 画热力图
def plot_heatmap(data, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=False, fmt=".2f", cmap='coolwarm', cbar=True)
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
sns.scatterplot(x=x_values, y=y_values, alpha=0.7, s=1)

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