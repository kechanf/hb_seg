from pylib.file_io import load_image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

final_recon_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
final_recon_list = pd.read_csv(final_recon_list_file)
# 有多少种病人？
patient_list = final_recon_list['patient_id'].unique()
print(patient_list)

mask_label_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/CerebrA_LabelDetails.csv"
mask_label = pd.read_csv(mask_label_file)
atlas_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/mni_icbm152_CerebrA_tal_nlin_sym_09c.v3draw"
atlas = load_image(atlas_file)[0]

seg_root = "/PBshare/SEU-ALLEN/Users/ZhixiYun/data/human_regi"
for dir in os.listdir(seg_root):
    if(not dir in patient_list):
        continue
    print(f"patient: {dir}")
    seg_mask_file = os.path.join(seg_root, dir, "Segmentation.seg_re1_global_nopadding_local.v3draw")
    seg_mask = load_image(seg_mask_file)[0]
    # print(np.unique(seg_mask))
    seg_mask = (seg_mask==2)

    # atlas_mip = np.max(atlas, axis=1)
    # seg_mask_mip = np.max(seg_mask, axis=1)

    current_atlas = atlas[seg_mask ==1]
    labels = np.unique(current_atlas)
    brain_regions = []
    for label in labels:
        rh_label_name = mask_label[mask_label['RH Label'] == label]['Label Name']
        lh_label_name = mask_label[mask_label['LH Labels'] == label]['Label Name']
        if(rh_label_name.empty == False):
            brain_regions.append((rh_label_name.values[0], "R"))
        if(lh_label_name.empty == False):
            brain_regions.append((lh_label_name.values[0], "L"))
    print(brain_regions)
