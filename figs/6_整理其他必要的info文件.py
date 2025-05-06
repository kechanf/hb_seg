import pandas as pd

meta_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
meta_info = pd.read_csv(meta_info_file, encoding="gbk")
pt_info = meta_info[["patient_number", "tissue_block_number"]]
# 去重
pt_info = pt_info.drop_duplicates()
print(pt_info.shape)

print(meta_info.shape)

# patient_info = meta_info[["patient_number", "gender", "age",
patient_info = {}

for _, row in meta_info.iterrows():
    patient_number = row["patient_number"]
    if patient_number not in patient_info:
        patient_info[patient_number] = {
            "age": row["age"],
            "gender": row["gender"],
            "brain_region": {row["brain_region"]},  # 使用集合去重
            "neuron_number": 1
        }
    else:
        patient_info[patient_number]["brain_region"].add(row["brain_region"])
        patient_info[patient_number]["neuron_number"] += 1

# 转换为列表，方便创建 DataFrame
data_list = [
    {
        "patient_number": pid,
        "age": info["age"],
        "gender": info["gender"],
        "brain_region": ";".join(info["brain_region"]),
        "neuron_number": info["neuron_number"]
    }
    for pid, info in patient_info.items()
]

df_patient = pd.DataFrame(data_list)
patient_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/patient_info.csv"
df_patient.to_csv(patient_info_file, index=False)
print(df_patient.shape)


tissue_info_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/tissue_info.csv"
tissue_info = {}

for _, row in meta_info.iterrows():
    tissue = f"{row['patient_number']}_{row['tissue_block_number']}"
    if tissue not in tissue_info:
        tissue_info[tissue] = {
            "patient_number": row["patient_number"],
            "tissue_block_number": row["tissue_block_number"],
            "age": row["age"],
            "gender": row["gender"],
            "brain_region": {row["brain_region"]},  # 使用集合自动去重
            "neuron_number": 1
        }
    else:
        tissue_info[tissue]["brain_region"].add(row["brain_region"])
        tissue_info[tissue]["neuron_number"] += 1

# 将 tissue_info 转换为列表，每个元素为一个字典
data_list = [
    {
        "patient_number": info["patient_number"],
        "tissue_block_number": info["tissue_block_number"],
        "age": info["age"],
        "gender": info["gender"],
        "brain_region": ";".join(info["brain_region"]),
        "neuron_number": info["neuron_number"]
    }
    for info in tissue_info.values()
]

tissue_info_df = pd.DataFrame(data_list, columns=["patient_number", "tissue_block_number", "age", "gender", "brain_region", "neuron_number"])
print(tissue_info_df.shape)
tissue_info_df.to_csv(tissue_info_file, index=False)