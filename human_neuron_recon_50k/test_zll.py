import os
import shutil
import pandas as pd


todo_id_list_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/lostImage.csv"
todo_id_list = pd.read_csv(todo_id_list_file, header=None)
todo_id_list = todo_id_list[0].tolist()
# print(todo_id_list)
save_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/to_LingliZhang/v3draw_0224"

v3draw_root = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"
v3draw_files, marker_files = [], []
# walk
for root, dirs, files in os.walk(v3draw_root):
    if("human_brain_data_v3draw" not in root):
        continue
    for file in files:
        current_id = str(int(file.split("_")[0]))
        if(current_id in todo_id_list):
            if file.endswith(".v3draw"):
                v3draw_files.append(os.path.join(root, file))
                shutil.copy(os.path.join(root, file), os.path.join(save_dir, file))
            elif file.endswith(".marker"):
                marker_files.append(os.path.join(root, file))
                shutil.copy(os.path.join(root, file), os.path.join(save_dir, file))



