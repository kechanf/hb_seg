import glob
import os
import subprocess
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def calc_global_features(swc_file, vaa3d=r'D:\Vaa3D_V4.001_Windows_MSVC_64bit\vaa3d_msvc.exe'):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x global_neuron_feature -f compute_feature -i "{swc_file}"'
    # cmd_str = f"{vaa3d} /x global_neuron_feature /f compute_feature /i {swc_file}"
    p = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    output_copy = output
    output = output.decode().splitlines()[35:-2]
    id = os.path.split(swc_file)[-1].split('_')[0]

    info_dict = {}
    for s in output:
        it1, it2 = s.split(':')
        it1 = it1.strip()
        it2 = it2.strip()
        if (it2 == '-1.#IND'):
            it2 = '-1'
        info_dict[it1] = float(it2)

    try:
        features = {
            'ID': id,
            'N_node': int(info_dict['N_node']),
            'Soma_surface': info_dict['Soma_surface'],
            'N_stem': int(info_dict['N_stem']),
            'Number of Bifurcatons': int(info_dict['Number of Bifurcatons']),
            'Number of Branches': int(info_dict['Number of Branches']),
            'Number of Tips': int(info_dict['Number of Tips']),
            'Overall Width': info_dict['Overall Width'],
            'Overall Height': info_dict['Overall Height'],
            'Overall Depth': info_dict['Overall Depth'],
            'Average Diameter': info_dict['Average Diameter'],
            'Total Length': info_dict['Total Length'],
            'Total Surface': info_dict['Total Surface'],
            'Total Volume': info_dict['Total Volume'],
            'Max Euclidean Distance': info_dict['Max Euclidean Distance'],
            'Max Path Distance': info_dict['Max Path Distance'],
            'Max Branch Order': info_dict['Max Branch Order'],
            'Average Contraction': info_dict['Average Contraction'],
            'Average Fragmentation': info_dict['Average Fragmentation'],
            'Average Parent-daughter Ratio': info_dict['Average Parent-daughter Ratio'],
            'Average Bifurcation Angle Local': info_dict['Average Bifurcation Angle Local'],
            'Average Bifurcation Angle Remote': info_dict['Average Bifurcation Angle Remote'],
            'Hausdorff Dimension': info_dict['Hausdorff Dimension']
        }
    except:
        print(f"FUCK at {swc_file}")
        print(cmd_str)
        print(output_copy)
        # time.sleep(1000)
        return None

    # print(features)
    return features


def plot_violin(df_gt, df_pred, violin_png):
    feature_name = ['N_node', 'Soma_surface', 'N_stem', 'Number of Bifurcatons',
                    'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
                    'Overall Depth', 'Average Diameter', 'Total Length', 'Total Surface',
                    'Total Volume', 'Max Euclidean Distance', 'Max Path Distance',
                    'Max Branch Order', 'Average Contraction', 'Average Fragmentation',
                    'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
                    'Average Bifurcation Angle Remote', 'Hausdorff Dimension']

    # plt.figure(figsize=(20, 20))

    num_features = len(feature_name)
    cols = 3  # 每行显示3个子图
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=False)
    axes = axes.flatten()

    df_gt['Type'] = 'GT'  # "ce+dice"
    df_pred['Type'] = 'Pred'  # "ce+dice+ptls"

    df = pd.concat([df_gt, df_pred], axis=0)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_name, var_name='Feature', value_name='Value')

    for idx, feature in enumerate(feature_name):
        ax = axes[idx]

        sns.violinplot(x='Feature', y='Value', hue='Type', data=df_long[df_long['Feature'] == feature], split=True,
                       ax=ax)
        ax.set_title(feature)
        ax.set_xlabel('')  # 清除x轴标签
        ax.set_ylabel('')  # 清除y轴标签
        ax.legend().set_visible(False)  # 在每个子图中隐藏图例

        if idx == 0:  # 只在第一个子图中显示图例
            ax.legend(title='Data Type', loc='upper right')

        # 隐藏空余的子图
    for ax in axes[num_features:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(violin_png)
    plt.close()


def process_files(gt_file, pred_file, v3d_path):
    features_gt = calc_global_features(gt_file, vaa3d=v3d_path)
    features_pred = calc_global_features(pred_file, vaa3d=v3d_path)
    if features_gt is not None and features_pred is not None:
        return (features_gt, features_pred)
    return None


def l_measure_gt_and_pred(gt_dir, pred_dir, gt_csv, pred_csv, violin_png,
                          v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x", debug=False):
    features_all = pd.DataFrame(columns=['ID', 'N_node', 'Soma_surface', 'N_stem', 'Number of Bifurcatons',
                                         'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
                                         'Overall Depth', 'Average Diameter', 'Total Length', 'Total Surface',
                                         'Total Volume', 'Max Euclidean Distance', 'Max Path Distance',
                                         'Max Branch Order', 'Average Contraction', 'Average Fragmentation',
                                         'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
                                         'Average Bifurcation Angle Remote', 'Hausdorff Dimension'])

    features_all.to_csv(gt_csv, float_format='%g', index=False)
    features_all.to_csv(pred_csv, float_format='%g', index=False)

    gt_files = glob.glob(os.path.join(gt_dir, '*swc'))
    pred_files = glob.glob(os.path.join(pred_dir, '*swc'))
    gt_files.sort()
    pred_files.sort()

    gt_ids = [int(os.path.split(f)[-1].split('_')[0]) for f in gt_files]
    pred_ids = [int(os.path.split(f)[-1].split('_')[0]) for f in pred_files]
    shared_ids = list(set(gt_ids) & set(pred_ids))

    # debug
    if (debug):
        shared_ids = shared_ids[:10]

    filtered_gt_files = [f for f, id in zip(gt_files, gt_ids) if id in shared_ids]
    filtered_pred_files = [f for f, id in zip(pred_files, pred_ids) if id in shared_ids]

    features_all_gt = []
    features_all_pred = []

    with ThreadPoolExecutor(max_workers=12) as executor:  # 可以根据你的系统调整 max_workers
        # 设置进度条
        progress_bar = tqdm(total=len(filtered_gt_files), desc='Processing_gt')

        # 提交任务到线程池
        future_to_files = {executor.submit(process_files, gt, pred, v3d_path): (gt, pred) for gt, pred in
                           zip(filtered_gt_files, filtered_pred_files)}

        # 处理线程池的结果
        for future in as_completed(future_to_files):
            result = future.result()
            if result is not None:
                features_gt, features_pred = result
                features_all_gt.append(features_gt)
                features_all_pred.append(features_pred)
            progress_bar.update(1)

    progress_bar.close()
    # print(features_all_gt)
    df_gt = pd.DataFrame(features_all_gt)
    df_gt.to_csv(gt_csv, float_format='%g', index=False, mode='a', header=False)

    df_pred = pd.DataFrame(features_all_pred)
    df_pred.to_csv(pred_csv, float_format='%g', index=False, mode='a', header=False)
    progress_bar.close()

    plot_violin(df_gt, df_pred, violin_png)


if __name__ == '__main__':
    # gt_dir = r"/data/kfchen/nnUNet/gt_swc"
    # pred_dir = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/10847_auto_v1.4_12k/swc"
    gt_dir = r"/data/kfchen/nnUNet/gt_swc"
    pred_dir = r"/data/kfchen/nnUNet/nnUNet_raw/result500_fb/v3dswc"

    gt_csv = r"/data/kfchen/nnUNet/gt_swc.csv"
    pred_csv = r"/data/kfchen/nnUNet/pred_swc.csv"
    violin_png = r"/data/kfchen/nnUNet/violin.png"
    v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"

    if (os.path.exists(pred_csv)):
        os.remove(pred_csv)
    if (os.path.exists(gt_csv)):
        os.remove(gt_csv)
    if (os.path.exists(violin_png)):
        os.remove(violin_png)

    l_measure_gt_and_pred(gt_dir, pred_dir, gt_csv, pred_csv, violin_png, v3d_path=v3d_path)
