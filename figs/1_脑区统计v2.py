import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


final_recon_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
final_recon_list = pd.read_csv(final_recon_list_file)
# 有多少种脑区？
recon_brain_regions = final_recon_list['brain_region'].unique()
print(recon_brain_regions)

brain_regions = {
    'frontal lobe': {
        "superior frontal gyrus":["SFG.R", "SFG.L", "SFG"],
        "middle frontal gyrus":["MFG.R", "MFG", "MFG.L"],
        "inferior frontal gyrus":["IFG", "IFG.R"],
        "frontal pole": ['FP.L', 'FP.R'],
        "ambiguous": ['FL.L', 'FL.R', '(X)FG', 'S(M)FG.R', 'M(I)FG.L', ], # 后面两个是交叉脑区

        "frontal tubercle": ['FT.L'], # 不在allen的atlas里面
    },
    'parietal lobe': {
        'inferior parietal gyrus': ["IPL", "IPL.L", 'IPL-near-AG'],
        "ambiguous": ["PL.L", "PL"],
    },
    'temporal lobe': {
        "superior temporal gyrus": ["STG.R", "STG", 'STG-AP'],
        "middle temporal gyrus": ["MTG.R", "MTG.L", "MTG"],
        "temporal pole": ["TP.R", "TP", "TP.L"],
        "ambiguous": ['TL.L', 'TL.R', 'S(M,I)TG', "S(M)TG.R", 'S(M)TG.L'] # 后面三个是交叉脑区
    },
    'occipital lobe': {
        "ambiguous": ['OL.L', 'OL.R']
    },
    "ambiguous":{
        "ambiguous": ["PL.L_OL.L"] # 不在allen的atlas里面
    }
}

CerebrA_atlas = {
'''

'''
    "superior frontal gyrus": 38,
    "middle frontal gyrus":[42, 1],
    "inferior frontal gyrus":0,
    "frontal pole": 0,

    "frontal tubercle":0,

    "inferior parietal gyrus": 0,

    "superior temporal gyrus": 45,
    "middle temporal gyrus":28,
    "temporal pole":0,



}

# unknown brain regions: {'S(M)TG.L', , 'S(M)TG.R',
# 'M(I)FG.L', 'FT.L', 'STG-AP', 'S(M,I)TG',
# 'S(M)FG.R', 'IPL-near-AG', , 'PL.L_OL.L'}
brain_region_mapping = {
    'frontal lobe': 'FL',
    'parietal lobe': 'PL',
    'temporal lobe': 'TL',
    'occipital lobe': 'OL',
}

def prepare_data():
    total_brain_regions = []
    for k, v in brain_regions.items():
        for k1, v1 in v.items():
            total_brain_regions.extend(v1)
    print(total_brain_regions)

    print(f"known brain regions: {set(recon_brain_regions) & set(total_brain_regions)}")
    # print(f"unknown brain regions: {set(recon_brain_regions) - set(total_brain_regions)}")
    unknown_brain_regions = list(set(recon_brain_regions) - set(total_brain_regions))
    print(f"unknown brain regions: {unknown_brain_regions}")

    patient_info_file = "/data/kfchen/trace_ws/patient_info.xlsx"
    patient_info = pd.read_excel(patient_info_file)
    unknown_brain_regions_info = patient_info[patient_info['english_abbr_nj'].isin(unknown_brain_regions)][["english_abbr_nj", "english_full_name", "intracranial_location"]]
    unique_unknown_brain_regions_info = unknown_brain_regions_info.drop_duplicates()

    # 完整显示
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(unique_unknown_brain_regions_info)

    final_recon_list_in_unknown_brain_regions = final_recon_list[final_recon_list['brain_region'].isin(unknown_brain_regions)]
    # discribe the number of neurons in each brain region
    print(final_recon_list_in_unknown_brain_regions['brain_region'].value_counts())

mask_mgz_file = "/data/kfchen/trace_ws/atlas/yale/YBA_696.nii"

img = nib.load(mask_mgz_file)

# 获取数据并转化为numpy数组
data = img.get_fdata()
print(data.shape)
print(data[10, 10, 10])
print(data.max(), data.min())
# hist
# plt.hist(data.flatten(), bins=100, range=(0, 1000))
# plt.ylim(1, 700)
#
# # 选择最大强度投影的轴（例如，选择z轴）
# mip = np.max(data, axis=2)
mip = data[60, :, :]
#
# # 绘制MIP图像
plt.imshow(mip.T, cmap='gray', origin='lower')
# cbar
plt.colorbar()

# plt.colorbar()
# plt.title('Maximum Intensity Projection (MIP) of MGZ File')
# plt.axis('off')
plt.show()

