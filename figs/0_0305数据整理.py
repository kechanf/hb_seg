import pandas as pd
import os
import shutil
from pylib.swc_handler import parse_swc, write_swc, crop_spheric_from_soma
from joblib import Parallel, delayed
from tqdm import tqdm
from simple_swc_tool.l_measure_api import l_measure_swc_dir

def work1(): # 重新合并一下两个l measure结果，6007 6008 2578 2796重新录入
    csv_files = ["train_val_list.csv", "test_list_with_gs.csv", "unlabeled_list.csv"]
    total_neuron_id_list = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", csv_file))
        current_id_list = df['id'].tolist()
        print(len(current_id_list))
        total_neuron_id_list.extend(current_id_list)
    print(len(total_neuron_id_list))
    # 1100 + 242 + 7502 = 8844

    l_measure_files = [
        "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc_l_measure.csv",
        "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/8_estimated_radius_swc_l_measure.csv"
    ]
    total_df = pd.DataFrame()
    for l_measure_file in l_measure_files:
        df = pd.read_csv(l_measure_file)
        df = df[df['ID'].isin(total_neuron_id_list)]
        total_df = pd.concat([total_df, df])
        # print(df[df['ID'] == 2578])

    # 删掉id不在total_neuron_id_list中的行
    # total_df = total_df[total_df['ID'].isin(total_neuron_id_list)]
    print(total_df.shape)
    # sort
    total_df = total_df.sort_values(by='ID')
    total_df.to_csv("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv", index=False)

    # 哪些缺失了
    missing_id_list = list(set(total_neuron_id_list) - set(total_df['ID'].tolist()))
    print(len(missing_id_list), missing_id_list)

def work2(): # 确认一下新老编号的映射问题
    l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv"
    l_measure_df_list = pd.read_csv(l_measure_file)[["ID"]].values.tolist()
    l_measure_df_list = [int(x[0]) for x in l_measure_df_list]
    print(len(l_measure_df_list))

    old_meta_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/Human_SingleCell_TrackingTable_20240712.csv"
    new_meta_file = "/data/kfchen/trace_ws/meta_hb_0305.xlsx"

    old_meta = pd.read_csv(old_meta_file, encoding='gbk')[["Cell ID", "病人编号", "组织块编号", "切片编号", "年龄"]]
    old_meta = old_meta.rename(columns={"Cell ID": "cell_id", "病人编号": "patient_number", "组织块编号": "tissue_block_number", "切片编号": "slice_number"})
    new_meta = pd.read_excel(new_meta_file)[["cell_id", "patient_number","tissue_block_number","slice_number"]]
    # 只处理前15000
    new_meta = new_meta[new_meta["cell_id"] < 20001]

    # 调整一下格式
    # for i in range(len(old_meta)):
    #     old_meta.iloc[i, 1] = old_meta.iloc[i, 1].replace("P", "P00")
    #     old_meta.iloc[i, 2] = old_meta.iloc[i, 2].replace("T", "T0")
    #     old_meta.iloc[i, 3] = old_meta.iloc[i, 3].replace("*", "")
    # drop_list = []
    # for i in range(len(old_meta)-1):
    #     if(old_meta.iloc[i, 1] == "P024" and not old_meta.iloc[i, 4] == 48):
    #         # 删除这一行
    #         # old_meta = old_meta.drop(i)
    #         drop_list.append(i)
    # old_meta = old_meta.drop(drop_list)
    # 批量处理
    mask = ~((old_meta.iloc[:, 1] == "P024") & (old_meta.iloc[:, 4] != 48))
    old_meta = old_meta[mask].reset_index(drop=True)
    old_meta = old_meta[["cell_id", "patient_number","tissue_block_number","slice_number"]]

    # 批量处理
    old_meta["cell_id"] = old_meta["cell_id"].apply(lambda x: int(x))
    old_meta["patient_number"] = old_meta["patient_number"].apply(lambda x: x.replace("P", "P00"))
    old_meta["tissue_block_number"] = old_meta["tissue_block_number"].apply(lambda x: x.replace("T", "T0"))
    old_meta["slice_number"] = old_meta["slice_number"].apply(lambda x: x.replace("*", ""))


    print(old_meta.head())
    print(new_meta.head())
    # exit()




    id_map_file = "/data/kfchen/trace_ws/paper_trace_result/id_map.xlsx"
    id_map_df = pd.read_excel(id_map_file)[["cell_id_backUp", "cell_id"]]
    # rename columns
    id_map_df = id_map_df.rename(columns={"cell_id_backUp": "old_id", "cell_id": "new_id"})
    # to map
    id_map = {}
    for i in range(len(id_map_df)):
        id_map[id_map_df.iloc[i, 1]] = id_map_df.iloc[i, 0] # new_id -> old_id

    new_meta["cell_id"] = new_meta["cell_id"].apply(lambda x: id_map[x])
    # 检查是否有重复id
    print(new_meta[new_meta["cell_id"].duplicated()])


    old_meta = old_meta[old_meta["cell_id"].isin(l_measure_df_list)]
    new_meta = new_meta[new_meta["cell_id"].isin(l_measure_df_list)]
    new_id_list = new_meta["cell_id"].tolist()
    old_meta = old_meta[old_meta["cell_id"].isin(new_id_list)]

    # 对比new_meta和old_meta是否完全一致
    new_meta = new_meta.sort_values(by="cell_id")
    old_meta = old_meta.sort_values(by="cell_id")

    print(new_meta.shape, old_meta.shape)
    for i in range(len(new_meta)):
        if (new_meta.iloc[i, 0] != old_meta.iloc[i, 0]) or (new_meta.iloc[i, 1] != old_meta.iloc[i, 1]) or (new_meta.iloc[i, 2] != old_meta.iloc[i, 2]) or (new_meta.iloc[i, 3] != old_meta.iloc[i, 3]):
            print(new_meta.iloc[i, 0], old_meta.iloc[i, 0])
            print(new_meta.iloc[i, 1:], old_meta.iloc[i, 1:])
            print("error")
            exit()


def work3(): # 获取最终的meta信息表
    l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv"
    l_measure_df_list = pd.read_csv(l_measure_file)[["ID"]].values.tolist()
    l_measure_df_list = [int(x[0]) for x in l_measure_df_list]
    print(len(l_measure_df_list))

    old_meta_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/Human_SingleCell_TrackingTable_20240712.csv"
    new_meta_file = "/data/kfchen/trace_ws/meta_hb_0305.xlsx"

    old_meta = pd.read_csv(old_meta_file, encoding='gbk')[["Cell ID", "病人编号", "年龄", "性别"]]
    old_meta = old_meta.rename(
        columns={"Cell ID": "cell_id", "年龄": "age", "性别": "gender"})
    new_meta = pd.read_excel(new_meta_file)
    # 只处理前15000
    new_meta = new_meta[new_meta["cell_id"] < 20001]

    mask = ~((old_meta.iloc[:, 1] == "P024") & (old_meta.iloc[:, 2] != 48))
    old_meta = old_meta[mask].reset_index(drop=True)

    # 批量处理
    old_meta["cell_id"] = old_meta["cell_id"].apply(lambda x: int(x))

    id_map_file = "/data/kfchen/trace_ws/paper_trace_result/id_map.xlsx"
    id_map_df = pd.read_excel(id_map_file)[["cell_id_backUp", "cell_id"]]
    # rename columns
    id_map_df = id_map_df.rename(columns={"cell_id_backUp": "old_id", "cell_id": "new_id"})
    # to map
    id_map = {}
    for i in range(len(id_map_df)):
        id_map[id_map_df.iloc[i, 1]] = id_map_df.iloc[i, 0]  # new_id -> old_id

    new_meta["cell_id"] = new_meta["cell_id"].apply(lambda x: id_map[x])
    # 检查是否有重复id
    print(new_meta[new_meta["cell_id"].duplicated()])

    old_meta = old_meta[old_meta["cell_id"].isin(l_measure_df_list)]
    new_meta = new_meta[new_meta["cell_id"].isin(l_measure_df_list)]
    new_id_list = new_meta["cell_id"].tolist()
    old_meta = old_meta[old_meta["cell_id"].isin(new_id_list)]

    new_meta = new_meta.sort_values(by="cell_id")
    old_meta = old_meta.sort_values(by="cell_id")

    # 新加三列
    new_meta["age"] = old_meta["age"].tolist()
    new_meta["gender"] = old_meta["gender"].tolist()


    # 改变文件编码格式


    # new_meta.to_csv("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/meta.csv", index=False)
    save_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/meta.csv"
    new_meta.to_csv(save_file, index=False, encoding='gbk')

    print(new_meta.shape)


def work4(): # 转一下文件编码格式
    save_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/meta.xlsx"
    new_meta = pd.read_excel(save_file)
    # save as utf-8
    new_meta.to_csv(save_file.replace(".xlsx", ".csv"), index=False, encoding='gbk')


def work5(): # 把swc文件转移过来
    origin_swc_dirs = [
        "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc",
        "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/8_estimated_radius_swc"
    ]
    target_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um"

    l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv"
    l_measure_df_list = pd.read_csv(l_measure_file)[["ID"]].values.tolist()
    l_measure_df_list = [int(x[0]) for x in l_measure_df_list]

    for origin_swc_dir in origin_swc_dirs:
        for root, dirs, files in os.walk(origin_swc_dir):
            for file in files:
                if file.endswith(".swc"):
                    file_id = int(file.split("_")[0])
                    if file_id in l_measure_df_list:
                        shutil.copy(os.path.join(root, file), os.path.join(target_dir, file))


def work6(): # 检查一下缺少了哪一个
    l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv"
    l_measure_df_list = pd.read_csv(l_measure_file)[["ID"]].values.tolist()
    l_measure_df_list = [int(x[0]) for x in l_measure_df_list]

    target_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um"
    swc_files = os.listdir(target_dir)

    print(len(swc_files), len(l_measure_df_list))

    swc_list = [int(f.split("_")[0]) for f in swc_files]
    missing_list = list(set(l_measure_df_list) - set(swc_list))
    print(len(missing_list), missing_list)

def work7(): # 转移crop后的swc
    l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv"
    l_measure_df_list = pd.read_csv(l_measure_file)[["ID"]].values.tolist()
    l_measure_df_list = [int(x[0]) for x in l_measure_df_list]


    source_dir = "/data/kfchen/trace_ws/cropped_swc/proposed_1um"
    target_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um_cropped_150um"
    os.makedirs(target_dir, exist_ok=True)

    swc_files = os.listdir(source_dir)
    for swc in swc_files:
        swc_id = int(swc.split("_")[0])
        if swc_id in l_measure_df_list:
            shutil.copy(os.path.join(source_dir, swc), os.path.join(target_dir, swc))

def work8():
    crop_radius = 150

    def current_task(swc_file, output_swc_file):
        df_tree = pd.read_csv(swc_file, comment='#', sep=' ', index_col=0,
                              names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))

        # cropping
        tree_out = crop_spheric_from_soma(df_tree, crop_radius)

        tree_out.to_csv(output_swc_file, sep=' ', index=True, header=False)

    source_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um"
    target_dir = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um_cropped_150um"
    os.makedirs(target_dir, exist_ok=True)

    swc_files = os.listdir(source_dir)
    Parallel(n_jobs=8)(delayed(current_task)(os.path.join(source_dir, swc), os.path.join(target_dir, swc)) for swc in tqdm(swc_files))

def check_neuron_regions(): # 统计脑区情况
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

    neuron_meta_14k_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
    neuron_meta_14k = pd.read_csv(neuron_meta_14k_file, encoding='gbk')
    recon_brain_region = neuron_meta_14k["brain_region"].tolist()
    # recon_brain_region = neuron_meta_14k[neuron_meta_14k['Cell ID'].isin(total_neuron_id_list)]['脑区'].tolist()
    recon_brain_region = list(set(recon_brain_region))

    # print(recon_brain_region)
    print(set(recon_brain_region).difference(set(exist_brain_region_labels)))
    print(set(exist_brain_region_labels).difference(set(recon_brain_region)))

def check_patient():
    meta_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
    meta_info = pd.read_csv(meta_info_file, encoding='gbk')

    age = meta_info[['patient_number', "age"]]
    unique_pairs = age.drop_duplicates()
    print("平均年龄", unique_pairs['age'].mean())

    meta_info = meta_info[['patient_number', "gender"]]
    print(len(meta_info))
    unique_pairs = meta_info.drop_duplicates()
    # 男性数量和女性数量
    print("男性数量：", len(unique_pairs[unique_pairs["gender"] == "Male"]), "百分比", len(unique_pairs[unique_pairs["gender"] == "Male"]) / len(unique_pairs))
    print("女性数量：", len(unique_pairs[unique_pairs["gender"] == "Female"]), "百分比", len(unique_pairs[unique_pairs["gender"] == "Female"]) / len(unique_pairs))
    print("病人总数：")
    print(f"unique_pairs(slice): {len(unique_pairs)}")
    dead_patient = ["P00002"]
    unique_pairs = unique_pairs[unique_pairs['patient_number'].isin(dead_patient)]
    print(f"dead_patient sample: {len(unique_pairs)}")


def check_slice():
    meta_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
    meta_info = pd.read_csv(meta_info_file, encoding='gbk')


    meta_info = meta_info[['patient_number', "tissue_block_number", "slice_number"]]
    print(len(meta_info))
    unique_pairs = meta_info.drop_duplicates()
    print("切片总数：")
    print(f"unique_pairs(slice): {len(unique_pairs)}")
    # print(meta_info)

    dead_patient = ["P00002"]
    unique_pairs = unique_pairs[unique_pairs['patient_number'].isin(dead_patient)]
    print(f"dead_patient sample: {len(unique_pairs)}")

def check_tissue():
    meta_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
    meta_info = pd.read_csv(meta_info_file, encoding='gbk')

    meta_info = meta_info[['patient_number', "tissue_block_number"]]
    print(len(meta_info))
    unique_pairs = meta_info.drop_duplicates()
    print("组织块总数：")
    print(f"unique_pairs(tissue): {len(unique_pairs)}")
    dead_patient = ["P00002"]
    unique_pairs = unique_pairs[unique_pairs['patient_number'].isin(dead_patient)]
    print(f"dead_patient sample: {len(unique_pairs)}")
    # print(meta_info)

def check_thickness():
    # 检查slice_thickness分布情况
    meta_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
    meta_info = pd.read_csv(meta_info_file, encoding='gbk')['slice_thickness']

    # 检查各种不同值的数量
    print(meta_info.value_counts())
    # 计算百分比
    print(meta_info.value_counts(normalize=True))

def check_ihc():
    meta_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
    meta_info = pd.read_csv(meta_info_file, encoding='gbk')['immunohistochemistry']

    print(meta_info.value_counts())
    # 计算百分比
    print(meta_info.value_counts(normalize=True))

# work1()
# work2()
# work3()
# work4()
# work5()
# work6()
# work7()
# work8()

# l_measure_swc_dir("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um_cropped_150um",
#                   "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um_cropped_150um_l_measure.csv",
#                   save_dir="/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/swc_1um_cropped_150um_l_measure_temp")

# check_neuron_regions() # 检查脑区情况

# check_patient()
# check_slice() # 检查切片情况
# check_tissue()

# check_thickness()
check_ihc()
