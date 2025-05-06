# 找到所有被重建的数据，然后
import os
import pandas as pd
from sympy.physics.units import percent
import seaborn as sns
import matplotlib.pyplot as plt
from simple_swc_tool.l_measure_api import l_measure_swc_dir



def get_final_id_list(final_recon_list_file = r"/data/kfchen/trace_ws/paper_trace_result/final_recon_list.csv"):

    if(not os.path.exists(final_recon_list_file)):
        recon_swc_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc"
        recon_swc_list = [f for f in os.listdir(recon_swc_dir) if f.endswith(".swc")]
        ids = [f.split("_")[0] for f in recon_swc_list]
        ids = [int(i) for i in ids]
        print(len(ids))

        good_sample_list_file = r"/data/kfchen/trace_ws/paper_trace_result/good_sample_list.csv"
        good_sample_list = pd.read_csv(good_sample_list_file)["id"].tolist()
        good_sample_ids = [int(i) for i in good_sample_list]
        print(len(good_sample_ids))

        muti_neuron_list_file = r"/data/kfchen/trace_ws/paper_trace_result/mutineuron_list.csv"
        muti_neuron_list = pd.read_csv(muti_neuron_list_file)["id"].tolist()
        muti_neuron_ids = [int(i) for i in muti_neuron_list]
        print(len(muti_neuron_ids))

        # in good_sample and not in muti_neuron
        recon_ids = list(set(ids).intersection(set(good_sample_ids)).difference(set(muti_neuron_ids)))
        recon_ids.sort()
        print(len(recon_ids))

        final_id_list = [str(i) for i in recon_ids]
        final_id_list_df = pd.DataFrame(final_id_list, columns=["id"])
        final_id_list_df.to_csv(final_recon_list_file, index=False)
    else:
        print("final_recon_list_file exists")
        final_id_list_df = pd.read_csv(final_recon_list_file)
        final_id_list = final_id_list_df["id"].tolist()
    print(f"final_id_list: {len(final_id_list)}")

    return final_id_list

def check_final_list(final_recon_list_file = r"/data/kfchen/trace_ws/paper_trace_result/final_recon_list.csv"):
    final_id_list_df = pd.read_csv(final_recon_list_file)
    final_id_list = final_id_list_df["id"].tolist()
    final_id_list = [int(i) for i in final_id_list]

    train_val_list_file = "/data/kfchen/trace_ws/paper_trace_result/train_val_list.csv"
    train_val_list = pd.read_csv(train_val_list_file)["id"].tolist()
    train_val_ids = [int(i) for i in train_val_list]

    test_list_file = "/data/kfchen/trace_ws/paper_trace_result/test_list_with_gs.csv"
    test_list = pd.read_csv(test_list_file)["id"].tolist()
    test_ids = [int(i) for i in test_list]

    unlabel_list_file = "/data/kfchen/trace_ws/paper_trace_result/test_list_without_gs.csv"
    unlabel_list = pd.read_csv(unlabel_list_file)["id"].tolist()
    unlabel_ids = [int(i) for i in unlabel_list]

    muti_neuron_list_file = r"/data/kfchen/trace_ws/paper_trace_result/mutineuron_list.csv"
    muti_neuron_list = pd.read_csv(muti_neuron_list_file)["id"].tolist()
    muti_neuron_ids = [int(i) for i in muti_neuron_list]



    set1 = set(final_id_list)
    set2 = set(test_ids) | set(train_val_ids) | set(unlabel_ids)
    print(len(set1))
    print(len(set2))

    # 检查不一样的
    print(set1.difference(set2))
    print(len(set1.difference(set2)))
    print(set2.difference(set1))
    print(len(set2.difference(set1)))

    print(len(train_val_ids), len(test_ids), len(unlabel_ids))

    """
    {6008, 2578, 2796, 6007}重建出来train_val有，但是final里面没有
    都不是多neuron
    6007 6008不是good sample, 2578 2796看起来是重建失败
    
    在good_sample中添加6007 6008
    
    
    在seg0中都有了，即都成功被预测
    
    train_val test是完全正确的，和多neuron不交
    
    应该ok了
    """
def get_unlabeled_list(unlabeled_recon_list_file = r"/data/kfchen/trace_ws/paper_trace_result/unlabeled_list.csv"):
    if(not os.path.exists(unlabeled_recon_list_file)):
        final_recon_list_file = r"/data/kfchen/trace_ws/paper_trace_result/final_recon_list.csv"
        final_id_list_df = pd.read_csv(final_recon_list_file)
        final_id_list = final_id_list_df["id"].tolist()
        final_id_list = [int(i) for i in final_id_list]

        train_val_list_file = "/data/kfchen/trace_ws/paper_trace_result/train_val_list.csv"
        train_val_list = pd.read_csv(train_val_list_file)["id"].tolist()
        train_val_ids = [int(i) for i in train_val_list]

        test_list_file = "/data/kfchen/trace_ws/paper_trace_result/test_list_with_gs.csv"
        test_list = pd.read_csv(test_list_file)["id"].tolist()
        test_ids = [int(i) for i in test_list]

        # final_id_list - train_val - test
        unlabel_list = list(set(final_id_list).difference(set(train_val_ids)).difference(set(test_ids)))
        unlabel_list.sort()
        print(len(unlabel_list))

        unlabel_list_id = [str(i) for i in unlabel_list]
        unlabel_list_df = pd.DataFrame(unlabel_list_id, columns=["id"])
        unlabel_list_df.to_csv(unlabeled_recon_list_file, index=False)


    else:
        print("unlabeled_recon_list_file exists")
        final_id_list_df = pd.read_csv(unlabeled_recon_list_file)
        final_id_list = final_id_list_df["id"].tolist()
    print(f"unlabeled_id_list: {len(final_id_list)}")

    return final_id_list

def get_new_neuron_info():
    def get_ids_from_csv(file_path=r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_list.csv"): # 从最终重建获取id
        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 假设 'id' 列包含样本 ID
        if 'id' in df.columns:
            ids = df['id'].tolist()
            ids = [int(i) for i in ids]
            # print(ids)
            print(f"{len(ids)} samples")
            # 是否有重复？没有重复
            # print((len(ids) == len(set(ids))))
            return ids

    def get_patient_and_tissue_info(ids, excel_file=r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/Human_SingleCell_TrackingTable_20240712.csv"):
        # 读取 Excel 文件
        df = pd.read_csv(excel_file, encoding='gbk')
        filtered_df = df[df['Cell ID'].isin(ids)][['Cell ID', '病人编号', '组织块编号', '切片厚度(微米)']]
        filtered_df.columns = ['id', 'patient_id', 'tissue_id', 'slice_thickness']

        filtered_df = filtered_df.groupby('id').first().reset_index()
        print(f"{len(filtered_df)} samples")
        return filtered_df

    def get_additional_info_from_excel(patient_tissue_df, excel_file=r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/sample_info10302024.xlsx"):
        df = pd.read_excel(excel_file)
        results = []
        for _, row in patient_tissue_df.iterrows():
            id = row['id']
            patient_id = row['patient_id']
            tissue_id = row['tissue_id']
            slice_thickness = row['slice_thickness']

            # 从 df 中找到匹配的病人编号和组织编号的行
            matched_row = df[(df['patient_number'] == patient_id) & (df['tissue_id'] == tissue_id)]
            # print(patient_id, tissue_id)
            if(len(matched_row) == 0):
                if(patient_id == 'P008' and tissue_id == 'T02'):
                    gender, age, brain_region = 'Female', 47, 'PL.L'
                elif(patient_id == 'P020' and tissue_id == 'T02'):
                    gender, age, brain_region = 'Female', 39, 'OL.R'
                else:
                    print(f"{id} {patient_id} {tissue_id} not found")
                    continue
            else:
                gender = matched_row['gender'].values[0]
                age = int(matched_row['patient_age'].values[0])
                brain_region = matched_row['english_abbr_nj'].values[0]
                if(gender == '男'):
                    gender = 'Male'
                elif(gender == '女'):
                    gender = 'Female'

            # 将结果存储到结果列表中
            results.append([id, patient_id, tissue_id, gender, age, brain_region, slice_thickness])

        # 将结果转化为 DataFrame
        final_df = pd.DataFrame(results, columns=['id', 'patient_id', 'tissue_id', 'gender', 'age', 'brain_region', 'slice_thickness'])
        return final_df

    def check_tissue(final_df):
        unique_tissue_brain_pairs = final_df[['tissue_id', 'patient_id']].drop_duplicates()
        # print(unique_tissue_brain_pairs)

        # 统计不同的组织编号与脑区的组合数
        num_unique_tissue_brain_pairs = unique_tissue_brain_pairs.shape[0]

        # 打印结果
        # print(f"样本中有 {num_unique_tissue_brain_pairs} 个不同的组织编号（与病人组合）。")
        print(f"组织块总数：{num_unique_tissue_brain_pairs}")
    ids = get_ids_from_csv()
    patient_tissue_df = get_patient_and_tissue_info(ids)
    final_df = get_additional_info_from_excel(patient_tissue_df)

    # sort
    final_df = final_df.sort_values(by=['id'])
    final_df_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"

    # print(final_df)
    unique_slice_thickness = final_df['slice_thickness'].unique()
    slice_thickness_count = final_df['slice_thickness'].value_counts()
    percentage = slice_thickness_count / slice_thickness_count.sum()
    sorted_slice_thickness_count = slice_thickness_count.sort_values(ascending=False)
    print("切片厚度统计（按数量排序）：")
    for thickness in sorted_slice_thickness_count.index:
        # 取得切片厚度的数量和百分比
        count = slice_thickness_count.get(thickness, 0)  # 防止某个厚度没有出现
        perc = percentage.get(thickness, 0)  # 防止某个厚度没有出现
        print(f"{thickness}: {count}, {perc * 100:.2f}%")

    check_tissue(final_df)

    print("dont save")
    return
    if(os.path.exists(final_df_file)):
        os.remove(final_df_file)
    final_df.to_csv(final_df_file, index=False)

    # sort by id
    final_df = final_df.sort_values(by=['id'])

    plt.figure(figsize=(10, 6))
    # show 切片厚度和编号的关系
    # sns.scatterplot(data=final_df, x='id', y='slice_thickness')
    # 折线图
    sns.lineplot(data=final_df, x='id', y='slice_thickness')
    plt.show()
    plt.close()

def get_total_length(final_df_file):
    final_df = pd.read_csv(final_df_file)

    l_measure_swc_file = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc_l_measure.csv"
    l_measure_df = pd.read_csv(l_measure_swc_file)

    final_df = final_df.merge(l_measure_df[['ID', 'Total Length']], left_on='id', right_on='ID', how='left')
    final_df = final_df.drop(columns=['ID'])

    final_df.to_csv(final_df_file, index=False)

def check_gender(final_df_file):
    final_df = pd.read_csv(final_df_file)
    # print(final_df.shape)
    patient_id = final_df['patient_id'].tolist()
    # print(len(patient_id))

    patient_id = list(set(patient_id))
    patient_id.sort()
    # print(len(patient_id))
    # print(patient_id)

    patient_info_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/sample_info10302024.xlsx"
    df = pd.read_excel(patient_info_file)
    male_patient, female_patient = [], []

    for i in patient_id:
        matched_row = df[df['patient_number'] == i]
        if(len(matched_row) == 0):
            print(f"{i} not found")

        if (len(matched_row) == 0):
            if (patient_id == 'P008'):
                gender = 'Female'
            elif (patient_id == 'P020'):
                gender = 'Female'
            else:
                print(f"{i} {patient_id} not found")
                continue
        else:
            gender = matched_row['gender'].values[0]

        if (gender == '男'):
            gender = 'Male'
        elif (gender == '女'):
            gender = 'Female'

        if (gender == 'Male'):
            male_patient.append(i)
        else:
            female_patient.append(i)

    # print(len(male_patient))
    # print(len(female_patient))
    print(f"男性病人: {len(male_patient)}， 占比： {len(male_patient) / len(patient_id) * 100:.2f}%")
    print(f"女性病人: {len(female_patient)}， 占比： {len(female_patient) / len(patient_id) * 100:.2f}%")
    print(f"总病人数: {len(patient_id)}")

def fuck_cc_bn():
    # train_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/train_val_list.csv"
    # test_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
    # train_list = pd.read_csv(train_list_file)['id'].tolist() + pd.read_csv(test_list_file)['id'].tolist()

    neuron_info_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
    neuron_info = pd.read_csv(neuron_info_file)
    # brain_regions = neuron_info[neuron_info['id'].isin(train_list)]['brain_region'].tolist()
    brain_regions = neuron_info[['id', 'brain_region']]

    # 初始化样本列表
    sample_count = {
        'CB_tonsil.L': [],
        'BN.L': [],
        'CC.L': [],
    }

    # 遍历每一行，统计样本编号
    for index, row in brain_regions.iterrows():
        brain_region = row['brain_region']
        id = row['id']

        if brain_region in sample_count:
            sample_count[brain_region].append(id)

    # 打印每个脑区的样本编号列表
    for region, ids in sample_count.items():
        print(f'{region}: {ids}')

    to_kill_ids = sample_count['CC.L'] + sample_count['BN.L'] + sample_count['CB_tonsil.L']
    # 整理文件
    print(f"to kill: {len(to_kill_ids)}")

    todo_files = [
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv",
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/train_val_list.csv",
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_list.csv",
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv",
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv",
    ]
    for file in todo_files:
        df = pd.read_csv(file)
        print(f"{file}: {df.shape}")
        df = df[~df['id'].isin(to_kill_ids)]
        print(f"{file}: {df.shape}")
        df.to_csv(file, index=False)

def check_slice():
    # meta_info_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
    ???
    meta_info = pd.read_excel(meta_info_file)

    total_recon_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
    total_recon_list = pd.read_csv(total_recon_list_file)['id'].tolist()

    meta_info = meta_info[meta_info['cell_id'].isin(total_recon_list)][['patient_number', "tissue_block_number", "slice_number"]]
    print(len(meta_info))
    unique_pairs = meta_info.drop_duplicates()
    print("切片总数：")
    print(f"unique_pairs(slice): {len(unique_pairs)}")
    # print(meta_info)

    dead_patient = ["P00002"]
    meta_info = meta_info[meta_info['patient_number'].isin(dead_patient)]
    print(f"dead_patient sample: {len(meta_info)}")

def check_tissue():
    # meta_info_file = "/data/kfchen/trace_ws/meta_hb_50114.xlsx"
    ???
    meta_info = pd.read_excel(meta_info_file)

    total_recon_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
    total_recon_list = pd.read_csv(total_recon_list_file)['id'].tolist()

    meta_info = meta_info[meta_info['cell_id'].isin(total_recon_list)][['patient_number', "tissue_block_number"]]
    print(len(meta_info))
    unique_pairs = meta_info.drop_duplicates()
    print("组织块总数：")
    print(f"unique_pairs(tissue): {len(unique_pairs)}")


def fuck_nk():
    final_df_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
    final_df = pd.read_csv(final_df_file)
    recon_patient_id = final_df['patient_id'].tolist()
    recon_patient_id = list(set(recon_patient_id))
    recon_patient_id = [int(f[1:]) for f in recon_patient_id]
    recon_patient_id = ["P" + str(f).zfill(5) for f in recon_patient_id]

    print(len(recon_patient_id))
    print(recon_patient_id)

    patient_info_file = "/data/kfchen/trace_ws/patient_info.xlsx"

    patient_info_df = pd.read_excel(patient_info_file)

    sample_id = patient_info_df[patient_info_df['patient_number'].isin(recon_patient_id)]['sample_id'].tolist()
    final_df['sample_id'] = final_df['patient_id'].apply(
        lambda x: patient_info_df.loc[patient_info_df['patient_number'] == ("P" + str(int(x[1:])).zfill(5)), 'sample_id'].values[0]
        if len(patient_info_df.loc[patient_info_df['patient_number'] == ("P" + str(int(x[1:])).zfill(5)), 'sample_id'].values) > 0
        else None)
    final_df['hospital'] = final_df['sample_id'].apply(lambda x: str(x).split("-")[1] if x is not None else None)
    #
    # hospital_list = [str(f.split("-")[1]) for f in sample_id]
    # print(hospital_list)
    # hospital_list = list(set(hospital_list))
    # print(len(hospital_list))
    # print(hospital_list)

    banned_hospital_list = ['NK']

    hostpital_count = final_df['hospital'].value_counts()
    print(hostpital_count)
    banned_list = final_df[final_df['hospital'].isin(banned_hospital_list)]
    print(f"banned: {banned_list}")

    train_val_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/train_val_list.csv"
    test_val_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
    train_list = pd.read_csv(train_val_file)['id'].tolist() + pd.read_csv(test_val_file)['id'].tolist()
    test_val = pd.read_csv(test_val_file)['id'].tolist()
    to_kill_ids = banned_list['id'].tolist()


    print(f"train test in banned: {len(set(train_list).intersection(set(banned_list['id'].tolist())))}")
    print(f"test in banned: {len(set(test_val).intersection(set(banned_list['id'].tolist())))}")

    #
    # banned_patient_id = patient_info_df[patient_info_df['sample_id'].str.contains('|'.join(banned_hospital_list))]['patient_number'].tolist()
    # banned_patient_id = list(set(banned_patient_id))
    # # recon_patient_id & banned_patient_id
    # banned_patient_id = list(set(recon_patient_id).intersection(set(banned_patient_id)))
    # print(len(banned_patient_id))
    # banned_patient_id = ["P" + str(int(f[1:])).zfill(3) for f in banned_patient_id]
    # print(f"banned_patient_id: {banned_patient_id}")
    #
    # banned_recon = final_df[final_df['patient_id'].isin(banned_patient_id)]
    # print(len(banned_recon))

    # NK: 233
    # JSP: 1343
    # JZ: 318

    todo_files = [
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv",
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/train_val_list.csv",
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_list.csv",
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv",
        "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv",
    ]
    for file in todo_files:
        df = pd.read_csv(file)
        print(f"{file}: {df.shape}")
        df = df[~df['id'].isin(to_kill_ids)]
        print(f"{file}: {df.shape}")
        df.to_csv(file, index=False)

    '''
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv: (242, 1)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv: (242, 1)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/train_val_list.csv: (1100, 1)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/train_val_list.csv: (1100, 1)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_list.csv: (8639, 1)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_list.csv: (8406, 1)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv: (7297, 1)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv: (7064, 1)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv: (8639, 8)
    /data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv: (8406, 8)
    '''

def add_long_examples():
    long_sample_mip_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/checked_recon_mip"
    long_sample_mip_files = [f for f in os.listdir(long_sample_mip_dir) if f.endswith(".png")]
    long_sample_ids = [f.split("_")[0] for f in long_sample_mip_files]
    long_sample_ids = [int(i) for i in long_sample_ids]

    long_sample_id_file = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/final_long_sample_id.csv"
    long_sample_id_df = pd.DataFrame(long_sample_ids, columns=["id"])
    long_sample_id_df.to_csv(long_sample_id_file, index=False)

    unlabel_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv"
    unlabel_list = pd.read_csv(unlabel_list_file)
    merge_list = pd.concat([unlabel_list, long_sample_id_df])
    merge_list.to_csv(unlabel_list_file, index=False)

# 检查是否有交集
def check_intersection():
    csv_files = ["train_val_list.csv", "test_list_with_gs.csv", "unlabeled_list.csv"]
    
    set_list = []
    for file in csv_files:
        df = pd.read_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", file))
        set_list.append(set(df['id'].tolist()))

    for i in range(len(set_list)):
        for j in range(i+1, len(set_list)):
            print(f"{csv_files[i]} & {csv_files[j]}: {len(set_list[i].intersection(set_list[j]))}")

def del_muti_neuron():
    muti_neuron_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/mutineuron_list.csv"
    muti_neuron_list = pd.read_csv(muti_neuron_list_file)['id'].tolist()
    csv_files = ["train_val_list.csv", "test_list_with_gs.csv", "unlabeled_list.csv"]

    for file in csv_files:
        df = pd.read_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", file))
        print(f"{file}: {df.shape}")
        df = df[~df['id'].isin(muti_neuron_list)]
        print(f"{file}: {df.shape}")
        df.to_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", file), index=False)

def del_nk_neuron():
    neuron_meta_14k_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/Human_SingleCell_TrackingTable_20240712.csv"
    neuron_meta_14k = pd.read_csv(neuron_meta_14k_file, encoding='gbk')
    if('Hostpital' not in neuron_meta_14k.columns):
        patient_info_file = "/data/kfchen/trace_ws/patient_info.xlsx"
        patient_info_df = pd.read_excel(patient_info_file)
        for i in range(len(neuron_meta_14k)):
            patient_number = neuron_meta_14k.loc[i, '病人编号']
            patient_number = "P" + str(int(patient_number[1:])).zfill(5)
            hostpital = patient_info_df[patient_info_df['patient_number'] == patient_number]['sample_id'].values[0].split("-")[1]
            neuron_meta_14k.loc[i, 'Hostpital'] = hostpital
        neuron_meta_14k.to_csv(neuron_meta_14k_file, index=False, encoding='gbk')

    nk_neuron_list = neuron_meta_14k[neuron_meta_14k['Hostpital'] == 'NK']['Cell ID'].tolist()

    csv_files = ["train_val_list.csv", "test_list_with_gs.csv", "unlabeled_list.csv"]

    for file in csv_files:
        df = pd.read_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", file))
        print(f"{file}: {df.shape}")
        df = df[~df['id'].isin(nk_neuron_list)]
        print(f"{file}: {df.shape}")
        df.to_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", file), index=False)

def del_non_cortex():
    non_cortex_label = ['pLV.L', 'BN.L']
    neuron_meta_14k_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/Human_SingleCell_TrackingTable_20240712.csv"
    neuron_meta_14k = pd.read_csv(neuron_meta_14k_file, encoding='gbk')
    non_cortex_neuron_list = neuron_meta_14k[neuron_meta_14k['脑区'].isin(non_cortex_label)]['Cell ID'].tolist()

    csv_files = ["train_val_list.csv", "test_list_with_gs.csv", "unlabeled_list.csv"]

    for file in csv_files:
        df = pd.read_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", file))
        print(f"{file}: {df.shape}")
        df = df[~df['id'].isin(non_cortex_neuron_list)]
        print(f"{file}: {df.shape}")
        df.to_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", file), index=False)

def check_neuron_regions():
    csv_files = ["train_val_list.csv", "test_list_with_gs.csv", "unlabeled_list.csv"]
    total_neuron_id_list = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", csv_file))
        current_id_list = df['id'].tolist()
        total_neuron_id_list.extend(current_id_list)

    print(len(total_neuron_id_list)) # 9164

    brain_regions = {
        'frontal': {
            "Frontal body": ["SFG.R", "SFG.L", "SFG"],  # 和superior frontal gyrus是同一个区域
            "Middle frontal": ["MFG.R", "MFG", "MFG.L"],
            "Inferior frontal": ["IFG", "IFG.R"],
            "Frontal pole": ['FP.L', 'FP.R'],
            "ambiguous (Frontal lobe)": ['FL.L', 'FL.R', '(X)FG', 'M(I)FG.L', 'S(M)FG.R', ],  # 后面两个是交叉脑区

            # "frontal tubercle": ['FT.L'], # 不在allen的atlas里面
        },
        'parietal': {
            # "superior parietal gyrus": ["SPG.R", "SPG.L", "SPG"], # 9
            # 'inferior parietal gyrus': ["IPL", "IPL.L", 'IPL-near-AG'], # 10
            'Supramarginal': ["IPL", "IPL.L", 'IPL-near-AG'],  # 指的应该是同一个区域
            "Parietal": ["PL.L", "PL"],  # # 16 13 31 51
        },
        'temporal': {
            "Temporal body": ["STG.R", "STG", 'STG-AP', "S(M)TG.R", 'S(M)TG.L', "MTG.R", "MTG.L", "MTG"],
            # superior temporal gyrus
            # "middle temporal gyrus": ["MTG.R", "MTG.L", "MTG"], # 28
            # "inferior temporal gyrus": [], # 3
            "Temporal pole": ["TP.R", "TP", "TP.L"],
            "ambiguous (Temporal lobe)": ['TL.L', 'TL.R', 'S(M,I)TG', ]  # 后面三个是交叉脑区
        },
        'occipital': {
            "Occipital": ['OL.L', 'OL.R']
        },
        "ambiguous": {
            "ambiguous": ["PL.L_OL.L", "FL_TL.L"]  # 不在allen的atlas里面
        }
    }
    exist_brain_region_labels = []
    for lobe in brain_regions.keys():
        for region in brain_regions[lobe].keys():
            exist_brain_region_labels.extend(brain_regions[lobe][region])
    # print(exist_brain_region_labels)

    neuron_meta_14k_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/Human_SingleCell_TrackingTable_20240712.csv"
    neuron_meta_14k = pd.read_csv(neuron_meta_14k_file, encoding='gbk')
    recon_brain_region = neuron_meta_14k[neuron_meta_14k['Cell ID'].isin(total_neuron_id_list)]['脑区'].tolist()
    recon_brain_region = list(set(recon_brain_region))

    # print(recon_brain_region)
    print(set(recon_brain_region).difference(set(exist_brain_region_labels)))
    print(set(exist_brain_region_labels).difference(set(recon_brain_region)))


def merge_age():
    neuron_meta_14k_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/Human_SingleCell_TrackingTable_20240712.csv"
    neuron_meta_14k = pd.read_csv(neuron_meta_14k_file, encoding='gbk')
    print(neuron_meta_14k.shape)

    # 找到编号一样的行
    # todo_id_list = [2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2412, 2413, 2414, 2415, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2558, 2559, 2560, 2561]
    todo_p = "P024"
    age = 49
    neuron_meta_14k.loc[neuron_meta_14k['病人编号'] == todo_p, '年龄'] = age
    # 合并完全相同的行
    neuron_meta_14k = neuron_meta_14k.groupby('Cell ID').first().reset_index()
    print(neuron_meta_14k.shape)

    neuron_meta_14k.to_csv(neuron_meta_14k_file, index=False, encoding='gbk')

def prepare_long_sample_l_measure():
    swc_dir = "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/8_estimated_radius_swc"
    result_file = swc_dir + "_l_measure.csv"
    l_measure_swc_dir(swc_dir, result_file)


if __name__ == "__main__":
    # add_long_examples()
    # prepare_long_sample_l_measure()
    # exit()

    check_intersection() # ok 检查各个子数据集是否有重合
    del_muti_neuron() # ok, 0 sample to be del 检查已选数据集中是否有多神经元的例子
    del_nk_neuron()
    del_non_cortex()

    check_neuron_regions()

    # 使用平均年龄，合并行
    merge_age()


    exit()

    # get_final_id_list()
    # get_unlabeled_list()
    # check_final_list()

    final_df_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
    check_gender(final_df_file)
    # 检查来自多少个病人、组织
    # check_fi  g1_info(final_df_file)
    # 检查切片厚度

    get_new_neuron_info()
    check_slice()
    check_tissue()
    # exit()
    # if(not os.path.exists(final_df_file)):
    #     get_new_neuron_info()
    #
    # get_total_length(final_df_file)







