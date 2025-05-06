import pandas as pd
import os

l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv"
l_measure_df = pd.read_csv(l_measure_file)

l_measure_df_t_3 = l_measure_df[l_measure_df["N_stem"] <= 3]
print(l_measure_df_t_3.shape)
kill_list_1 = l_measure_df_t_3[l_measure_df_t_3["N_stem"] <=  1]["ID"].tolist()
print(len(kill_list_1))

# l_measure_df_t_2 = l_measure_df[l_measure_df["N_stem"] <= 2]
# print(l_measure_df_t_2.shape)

train_val_set_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/train_val_list.csv"
train_val_list = pd.read_csv(train_val_set_file)["id"].tolist()
test_set_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/test_list_with_gs.csv"
test_list = pd.read_csv(test_set_file)["id"].tolist()
important_list = train_val_list + test_list
print("total_len of train test val", len(important_list))
print("n stem <= 3", l_measure_df_t_3[l_measure_df_t_3["ID"].isin(important_list)].shape)
# print("n stem <= 2", l_measure_df_t_2[l_measure_df_t_2["ID"].isin(important_list)].shape)


# 找到total legth的平均值
print("total_length", l_measure_df["Total Length"].mean(), l_measure_df["Total Length"].std())
l_measure_df_t_50 = l_measure_df[l_measure_df["Total Length"] <= 200]
print("total_length <= 500", l_measure_df_t_50.shape)

# 找到4分位数
print("total_length <= 500", l_measure_df["Total Length"].quantile(0.05))
per5_t = l_measure_df["Total Length"].quantile(0.05)
l_measure_df_t_per5 = l_measure_df[l_measure_df["Total Length"] <= per5_t]
print("total_length <= 5%", l_measure_df_t_per5.shape)
kill_list_2 = l_measure_df_t_per5["ID"].tolist()

kill_list = list(set(kill_list_1 + kill_list_2))
print("kill_list", len(kill_list))
# if in train_val_list
a_kill_list = list(set(kill_list) & set(train_val_list))
print("a_kill_list in train_val_list", len(a_kill_list))

# if in test_list
b_kill_list = list(set(kill_list) & set(test_list))
print("b_kill_list in test_list", len(b_kill_list))














l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv"
l_measure_df = pd.read_csv(l_measure_file)
l_measure_df  = l_measure_df[~l_measure_df["ID"].isin(kill_list)]
print(l_measure_df.shape)
l_measure_df.to_csv(l_measure_file, index=False)

# crop lmeasure
l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_cropped_150um_l_measure.csv"
l_measure_df = pd.read_csv(l_measure_file)
l_measure_df  = l_measure_df[~l_measure_df["ID"].isin(kill_list)]
print(l_measure_df.shape)
l_measure_df.to_csv(l_measure_file, index=False)


meta_file = "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv"
meta_df = pd.read_csv(meta_file, encoding="gbk")
meta_df  = meta_df[~meta_df["cell_id"].isin(kill_list)]
print(meta_df.shape)
meta_df.to_csv(meta_file, index=False, encoding="gbk")

todo_dirs = [
    "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_cropped_150um",
    "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um",
    "/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/mip"
]
for todo_dir in todo_dirs:
    for file in os.listdir(todo_dir):
        curr_id = int(file.split("_")[0].split(".")[0])
        if curr_id in kill_list:
            os.remove(os.path.join(todo_dir, file))
    print("len of files in", todo_dir, len(os.listdir(todo_dir)))




