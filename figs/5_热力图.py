import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from neurom.features.morphology import feature
from scipy.spatial.distance import squareform
from skbio.stats.distance import mantel
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import fcluster

import seaborn as sns
from scipy import stats

# 设置清晰度
plt.rcParams['figure.dpi'] = 300

feature_name_mapping = {
    'N_stem': 'No. of Stems',
    'Number of Bifurcatons': 'No. of Bifurcations',
    'Number of Branches': 'No. of Branches',
    'Number of Tips': 'No. of Tips',
    'Overall Width': 'Width',
    'Overall Height': 'Height',
    'Overall Depth': 'Depth',
    'Total Length': 'Length',
    'Max Euclidean Distance': 'Max Euclidean',
    'Max Path Distance': 'Max path Dist.',
    'Max Branch Order': 'Max Branch Order',
}

def plot_heatmap(corr_matrix, title):
    # 3. 可视化 - 热力图
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)
    plt.show()
    plt.close()


def compare_heat_map(manual_corr, auto_corr, comp_corr, feature_name_mapping):
    # 设置颜色映射的范围（确保两图一致）
    vmin = min(manual_corr.min().min(), auto_corr.min().min())
    vmax = max(manual_corr.max().max(), auto_corr.max().max())

    fig, axes = plt.subplots(2, 1, figsize=(6, 14), constrained_layout=True, dpi=300)

    # 绘制人工标注的相关性热力图
    sns.heatmap(manual_corr, ax=axes[0], cmap='coolwarm', annot=False, fmt=".2f",
                cbar_kws={'shrink': 0.8}, vmin=vmin, vmax=vmax, cbar=False)
    axes[0].set_title('Manual Annotation', fontsize=24)

    # 获取当前的x和y轴标签，并根据mapping进行缩写
    current_x_labels = [feature_name_mapping.get(label, label) for label in manual_corr.columns]
    current_y_labels = [feature_name_mapping.get(label, label) for label in manual_corr.index]

    axes[0].set_xticklabels(current_x_labels, rotation=0, ha='right', fontsize=16)
    axes[0].set_yticklabels(current_y_labels, fontsize=16)
    axes[0].tick_params(axis='x', rotation=90, labelsize=16)
    axes[0].tick_params(axis='y', rotation=0, labelsize=16)

    # # 绘制自动处理的相关性热力图
    # sns.heatmap(auto_corr, ax=axes[1], cmap='coolwarm', annot=False, fmt=".2f",
    #             cbar_kws={'shrink': 0.8}, vmin=vmin, vmax=vmax, cbar=False)
    # axes[1].set_title('Auto Annotation', fontsize=24)
    #
    # # 获取当前的x和y轴标签，并根据mapping进行缩写
    # current_x_labels = [feature_name_mapping.get(label, label) for label in auto_corr.columns]
    # current_y_labels = [feature_name_mapping.get(label, label) for label in auto_corr.index]
    #
    # axes[1].set_xticklabels(current_x_labels, rotation=0, ha='right', fontsize=16)
    # axes[1].set_yticklabels(current_y_labels, fontsize=16)
    # axes[1].tick_params(axis='x', rotation=90, labelsize=16)
    # axes[1].set_yticks([])  # 不显示y轴刻度

    sns.heatmap(auto_corr, ax=axes[1], cmap='coolwarm', annot=False, fmt=".2f",
                cbar_kws={'shrink': 0.8}, vmin=vmin, vmax=vmax, cbar=False)



    # 绘制对比的相关性热力图
    sns.heatmap(comp_corr, ax=axes[1], cmap='coolwarm', annot=False, fmt=".2f",
                cbar_kws={'shrink': 0.8, 'orientation': 'horizontal'},
                vmin=vmin, vmax=vmax)
    axes[1].set_title('Comparison', fontsize=24)

    # 获取当前的x和y轴标签，并根据mapping进行缩写
    current_x_labels = [feature_name_mapping.get(label, label) for label in comp_corr.columns]
    current_y_labels = [feature_name_mapping.get(label, label) for label in comp_corr.index]

    axes[1].set_xticklabels(current_x_labels, rotation=0, ha='right', fontsize=16)
    axes[1].set_yticklabels(current_y_labels, fontsize=16)
    axes[1].tick_params(axis='x', rotation=90, labelsize=16)
    axes[1].tick_params(axis='y', rotation=0, labelsize=16)



    # 保存图像
    # plt.show()
    plt.savefig('/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/heatmap.png')
    plt.close()


if __name__ == '__main__':
    # mutineuron_list_file = "/data/kfchen/trace_ws/paper_trace_result/mutineuron_list.csv"
    # mutineuron_list = pd.read_csv(mutineuron_list_file)['id'].tolist()
    train_val_list_file = "/data/kfchen/trace_ws/paper_trace_result/train_val_list.csv"
    train_val_list = pd.read_csv(train_val_list_file)['id'].tolist()
    # total_recon_list = pd.read_csv("/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_list.csv")['id'].tolist()

    important_feasures = ['N_stem', 'Number of Branches', 'Number of Tips', 'Total Length']
    meaningful_feasures = ['N_stem', 'Number of Bifurcatons', 'Number of Branches', 'Number of Tips',
                           'Overall Width', 'Overall Height', 'Overall Depth', 'Total Length',
                           'Max Euclidean Distance', 'Max Path Distance', 'Max Branch Order',
                           ]
    other_feasures = ['N_node', 'Soma_surface',  'Average Diameter', 'Total Surface', 'Total Volume',
                      'Average Contraction', 'Average Fragmentation', 'Average Parent-daughter Ratio',
                      'Average Bifurcation Angle Local', 'Hausdorff Dimension', 'Average Bifurcation Angle Remote']

    l_measure_result_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv"
    # 读取 CSV 文件
    data = pd.read_csv(l_measure_result_file)
    filtered_data = data[meaningful_feasures]
    # features = filtered_data.drop(columns=['ID'])
    manual_corr = filtered_data.corr()
    data1 = filtered_data

    l_measure_result_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv"
    data = pd.read_csv(l_measure_result_file)
    filtered_data = data[(~data['ID'].isin(train_val_list))]
    filtered_data = filtered_data[meaningful_feasures]
    # print(len(filtered_data))
    # features = filtered_data.drop(columns=['ID'])
    auto_corr = filtered_data.corr()
    data2 = filtered_data

    # data1 和data2的相关性
    print(data1.shape, data2.shape)
    columns = [f'Feature_{i + 1}' for i in range(11)]
    df1 = pd.DataFrame(data1, columns=columns)
    df2 = pd.DataFrame(data2, columns=columns)

    print(df1.shape, df2.shape)
    print("mmanual_corr: ", manual_corr.shape, "auto_corr: ", auto_corr.shape)

    # 提取上三角的值（不包括对角线）
    corr_matrix1, corr_matrix2 = manual_corr.values, auto_corr.values

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
    # 范围 x:-0.2-1.2
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)

    # # 手动计算预测区间
    # x_pred = np.linspace(min(x_values), max(x_values), 100)
    # X = sm.add_constant(x_values)  # 添加截距项
    # model = sm.OLS(y_values, X).fit()
    # pred = model.get_prediction(sm.add_constant(x_pred))
    # pred_mean = pred.predicted_mean
    # pred_ci = pred.conf_int(alpha=0.05)  # 95% 预测区间
    #
    # # 绘制预测区间
    # plt.fill_between(x_pred, pred_ci[:, 0], pred_ci[:, 1],
    #                  color='blue', alpha=0.2, label='Prediction Interval (95%)')

    # 计算Pearson相关系数和p值
    r, p = stats.pearsonr(x_values, y_values)

    # 添加统计信息
    # plt.annotate(f'Pearson r = {r:.3f}\np-value = {p:.4f}',
    #              xy=(0.05, 0.95), xycoords='axes fraction',
    #              bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    # Pearson r = 0.958

    # 设置图形属性
    # plt.title('Scatter Plot of Upper Triangle Correlation Values', pad=20)
    plt.xlabel('Manual', fontsize=16)
    plt.ylabel('Auto', fontsize=16)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.axhline(0, color='black', lw=0.5)
    # plt.axvline(0, color='black', lw=0.5)

    plt.tight_layout()
    # plt.show()
    plt.savefig('/home/kfchen/neuron_seg_human/fig_results/fig6c_scatter_plot.png')

    exit()

    n = df2.shape[0] - df1.shape[0]  # 计算需要复制的样本数
    df1_expanded = df1.sample(n=n, replace=True, random_state=42)
    df1_expanded = pd.concat([df1, df1_expanded], axis=0, ignore_index=True)
    print(df1_expanded.shape, df2.shape)

    comp_corr = np.corrcoef(df1_expanded.values.T, df2.values.T)[0:11, 11:22]
    # 和manual_corr的index和columns一致
    comp_corr = pd.DataFrame(comp_corr, index=manual_corr.index, columns=manual_corr.columns)
    # plot_heatmap(auto_corr, 'Auto_anno')

    print(manual_corr.shape, auto_corr.shape, comp_corr.shape)
    compare_heat_map(manual_corr, auto_corr, comp_corr, feature_name_mapping)

    # diff_corr = manual_corr - auto_corr
    # # 计算平均绝对差值
    # mean_abs_diff = np.mean(np.abs(diff_corr.values.flatten()))
    # print(f'平均绝对差值: {mean_abs_diff}')
    # # 计算均方根误差
    # rmse = np.sqrt(np.mean((diff_corr.values.flatten()) ** 2))
    # print(f'均方根误差: {rmse}')
    #
    #
    # manual_flat = manual_corr.values.flatten()
    # auto_flat = auto_corr.values.flatten()
    # # 计算相关系数
    # corr_between_matrices = np.corrcoef(manual_flat, auto_flat)[0, 1]
    # print(f'两个相关性矩阵之间的相关系数: {corr_between_matrices}')
    #
    #
    # manual_dist = 1 - np.abs(manual_corr)
    # auto_dist = 1 - np.abs(auto_corr)
    # # 进行Mantel检验
    # statistic, p_value, n = mantel(squareform(manual_dist), squareform(auto_dist), method='pearson', permutations=999)
    # print(f'Mantel检验统计量: {statistic}, p值: {p_value}')
    #
    #
    # # 设置随机种子
    # seed = 42
    # # 对人工标注结果进行MDS
    # mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
    # manual_mds = mds.fit_transform(1 - manual_corr)
    # # 对自动处理结果进行MDS
    # auto_mds = mds.fit_transform(1 - auto_corr)
    # # 绘制MDS结果
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].scatter(manual_mds[:, 0], manual_mds[:, 1])
    # axes[0].set_title('人工标注结果的MDS')
    # axes[1].scatter(auto_mds[:, 0], auto_mds[:, 1])
    # axes[1].set_title('自动处理结果的MDS')
    # plt.show()
    # plt.close()
    #
    #
    #
    # # 对人工标注结果聚类
    # manual_linkage = linkage(manual_corr, method='average')
    # plt.figure(figsize=(8, 6))
    # dendrogram(manual_linkage, labels=manual_corr.columns)
    # plt.title('人工标注结果的聚类树')
    # plt.show()
    # # 对自动处理结果聚类
    # auto_linkage = linkage(auto_corr, method='average')
    # plt.figure(figsize=(8, 6))
    # dendrogram(auto_linkage, labels=auto_corr.columns)
    # plt.title('自动处理结果的聚类树')
    # plt.show()
    #
    #
    #
    # manual_labels = fcluster(manual_linkage, t=5, criterion='maxclust')
    # auto_labels = fcluster(auto_linkage, t=5, criterion='maxclust')
    # # 计算ARI
    # ari = adjusted_rand_score(manual_labels, auto_labels)
    # print(f'Adjusted Rand Index (ARI): {ari}')

    '''
    平均绝对差值: 0.13661615688188364
    均方根误差: 0.1979792632812942
    两个相关性矩阵之间的相关系数: 0.8553338278217363
    Mantel检验统计量: 0.8485404328713803, p值: 0.001
    Adjusted Rand Index (ARI): 0.5583849506767607

    '''







