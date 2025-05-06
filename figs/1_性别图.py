import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_stacked_age_gender_distribution(csv_file, traced_csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file, encoding='gbk')
    traced_df = pd.read_csv(traced_csv_file, encoding='gbk')

    ids = df['Cell ID']
    traced_ids = traced_df['ID']
    shared_ids = set(ids) & set(traced_ids)

    df = df[df['Cell ID'].isin(shared_ids)]


    # 创建年龄区间，这里以10年为一个区间进行分组
    bins = range(int(df['年龄'].min()), int(df['年龄'].max()) + 20, 10)
    labels = [f'{i}-{i + 9}' for i in bins[:-1]]
    df['AgeGroup'] = pd.cut(df['年龄'], bins=bins, labels=labels, right=False)

    # 按年龄分组并按性别计数
    age_gender_distribution = df.groupby(['AgeGroup', '性别']).size().unstack(fill_value=0)

    # 设定颜色代码
    colors = ['#F7B7D2', '#B8E5FA']  # 蓝色男性，粉色女性

    # 绘制堆叠柱状图
    # 向左对齐，以便在柱状图上添加数值标签
    ax = age_gender_distribution.plot(kind='bar', stacked=True, figsize=(4, 4), color=colors, width=-0.8, align='edge')
    ax.set_title('Age and Gender Distribution (Neuron)')
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Number of Neurons', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.tight_layout()

    # 调整条形间的间距
    for container in ax.containers:
        plt.setp(container, width=0.8)  # 设置条形图中条的宽度

    # 在柱状图上添加数值标签
    for i, (p, q) in enumerate(zip(ax.containers[0], ax.containers[1])):

        # 获取柱子的中心位置
        p_center = p.xy[0] + p.get_width() / 2
        q_center = q.xy[0] + q.get_width() / 2

        if (i > len(labels) - 3):
            ax.text(p_center, p._height / 2 + 100, '2\n0', ha='center', va='center', fontsize=10, color='black')
            break

        # 为每个柱子添加标签，水平和垂直都居中
        ax.text(p_center, p._height / 2, str(int(p._height)), ha='center', va='center', fontsize=10, color='black')
        ax.text(q_center, p._height + q._height / 2, str(int(q._height)), ha='center', va='center', fontsize=10,
                color='black')

    # 显示图表
    plt.show()

def plot_stacked_age_gender_distribution_patient(csv_file, traced_csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file, encoding='gbk')
    traced_df = pd.read_csv(traced_csv_file, encoding='gbk')

    ids = df['Cell ID']
    traced_ids = traced_df['ID']
    shared_ids = set(ids) & set(traced_ids)

    df = df[df['Cell ID'].isin(shared_ids)]

    unique_patients = df.drop_duplicates(subset=['病人编号'])

    # 创建年龄区间，这里以10年为一个区间进行分组
    bins = range(int(unique_patients['年龄'].min()), int(unique_patients['年龄'].max()) + 20, 10)
    labels = [f'{i}-{i + 9}' for i in bins[:-1]]
    unique_patients['AgeGroup'] = pd.cut(unique_patients['年龄'], bins=bins, labels=labels, right=False)

    # 按年龄分组并按性别计数
    # age_distribution = unique_patients.groupby(['AgeGroup']).size()
    age_gender_distribution = unique_patients.groupby(['AgeGroup', '性别']).size().unstack(fill_value=0)

    # 设定颜色代码
    colors = ['#F7B7D2', '#B8E5FA']  # 蓝色男性，粉色女性

    # 绘制堆叠柱状图
    # 向左对齐，以便在柱状图上添加数值标签
    ax = age_gender_distribution.plot(kind='bar', stacked=True, figsize=(5, 5), color=colors, width=-0.8, align='edge')
    ax.set_title('Age and Gender Distribution (Patient)')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Number of Patients')
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.tight_layout()

    # 调整条形间的间距
    for container in ax.containers:
        plt.setp(container, width=0.6)  # 设置条形图中条的宽度

    # 在柱状图上添加数值标签
    for i, (p, q) in enumerate(zip(ax.containers[0], ax.containers[1])):

        # 获取柱子的中心位置
        p_center = p.xy[0] + p.get_width() / 2
        q_center = q.xy[0] + q.get_width() / 2

        if (i > len(labels) - 3):
            ax.text(p_center, p._height / 2 + 100, '2\n0', ha='center', va='center', fontsize=9, color='black')
            break

        # 为每个柱子添加标签，水平和垂直都居中
        ax.text(p_center, p._height / 2, str(int(p._height)), ha='center', va='center', fontsize=9, color='black')
        ax.text(q_center, p._height + q._height / 2, str(int(q._height)), ha='center', va='center', fontsize=9,
                color='black')

    # 显示图表
    plt.show()


def plot_stacked_age_gender_distribution2(csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file, encoding='gbk')
    print(len(df))

    # 检查总的性别分布
    male_list = df[df['gender'] == 'Male']['id'].tolist()
    female_list = df[df['gender'] == 'Female']['id'].tolist()
    # print(f"male: {len(male_list)}, f: {len(female_list)}")
    # print(len(male_list)/len(df), len(female_list)/len(df), )
    print(f"male: {len(male_list)}, ratio: {len(male_list) / len(df) * 100:.2f}%")
    print(f"male: {len(female_list)}, ratio: {len(female_list) / len(df) * 100:.2f}%")


    # 创建年龄区间，这里以10年为一个区间进行分组
    bins = range(int(df['age'].min()), int(df['age'].max()) + 10, 10)
    labels = [f'{i}-{i + 9}' for i in bins[:-1]]
    df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    min_age = df['age'].min()
    max_age = df['age'].max()
    total_samples = len(df)
    male_samples = len(df[df['gender'] == 'Male'])  # 假设性别列中男性标记为'男性'
    male_ratio = male_samples / total_samples * 100  # 以百分比表示
    # print(min_age, max_age)
    print(f"max age: {max_age}, min age: {min_age}, mean age: {df['age'].mean()}")

    # 按年龄分组并按性别计数
    age_gender_distribution = df.groupby(['AgeGroup', 'gender']).size().unstack(fill_value=0)

    # 设定颜色代码
    colors = ['#F7B7D2', '#B8E5FA']  # 蓝色男性，粉色女性

    # 设置清晰度
    plt.rcParams['savefig.dpi'] = 300

    # 绘制堆叠柱状图
    # 向左对齐，以便在柱状图上添加数值标签
    ax = age_gender_distribution.plot(kind='bar', stacked=True, figsize=(6+1, 4+1), color=colors, align='edge', edgecolor='black', linewidth=0.5)
    # ax.set_title('Age and Gender Distribution (Neuron)')
    ax.set_xlabel('Age Group', fontsize=15)
    ax.set_ylabel('Number of Neurons', fontsize=15)
    plt.xticks(rotation=45, ha='center', x=0.05)
    # plt.legend(frameon=False, fontsize=12, loc='best')
    ax.legend(frameon=False, fontsize=15, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    plt.tick_params(axis='both', which='major', labelsize=15)  # 调整刻度标签大小
    plt.tight_layout()
    # 调整条形间的间距
    for container in ax.containers:
        plt.setp(container, width=0.5)  # 设置条形图中条的宽度


    # 关闭上面和右边的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # x轴刻度标签向右偏移


    # 在柱状图上添加数值标签
    # for i, (p, q) in enumerate(zip(ax.containers[0], ax.containers[1])):
    #
    #     # 获取柱子的中心位置
    #     p_center = p.xy[0] + p.get_width() / 2
    #     q_center = q.xy[0] + q.get_width() / 2

        # if (i > len(labels) - 3):
        #     ax.text(p_center, p._height / 2 + 100, '2\n0', ha='center', va='center', fontsize=10, color='black')
        #     break

        # # 为每个柱子添加标签，水平和垂直都居中
        # if(not int(p._height) == 0):
        #     # ax.text(p_center, p._height / 2, str(int(p._height)), ha='center', va='center', fontsize=15, color='black')
        #     # ax.text(q_center, p._height + q._height / 2, str(int(q._height)), ha='center', va='center', fontsize=15,
        #     #         color='black')
        #     number_list.append(p._height)
        #     number_list.append(q._height)

    plt.subplots_adjust(bottom=0.3)
    # 显示图表
    # plt.show()
    plt.savefig("/data/kfchen/trace_ws/paper_trace_result/age_gender_distribution.png")
    plt.close()
# 指定 CSV 文件路径
# csv_file = r"D:\tracing_ws\new_Human_SingleCell_TrackingTable_20240712.csv"
neuron_info_file = r"/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
plot_stacked_age_gender_distribution2(neuron_info_file)