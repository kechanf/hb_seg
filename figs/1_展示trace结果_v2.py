import pandas as pd
from mpl_toolkits.mplot3d.proj3d import transform

from pylib.file_io import load_image
import numpy as np
import os
from skimage.draw import line_aa
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# brain_regions = ["superior frontal gyrus", "middle frontal gyrus", "inferior frontal gyrus",
#             "parietal lobe", "inferior parietal lobe",
#             "superior temporal gyrus", "middle temporal gyrus",
#             'occipital lobe',
#             'temporal lobe',
#             'frontal lobe',
#             'others'
#             ]
# brain_region_map = {
#     'frontal lobe': ['FL.L', 'FL.R', 'FL_TL.L'],
#     "superior frontal gyrus": ["SFG.R", "SFG.L", "SFG", "S(M)FG.R", "M(I)FG.L"],
#     "middle frontal gyrus": ["MFG.R", "MFG", "MFG.L"],
#     "inferior frontal gyrus": ["IFG.R", "(X)FG", "IFG"],
#     #
#     'temporal lobe': ['TL.L', 'TL.R'],
#     "superior temporal gyrus": ["STG", "STG.R", "S(M)TG.R", "S(M)TG.L", "STG-AP", "S(M,I)TG"],
#     "middle temporal gyrus": ["MTG.R", "MTG.L", "MTG"],
#
#     "parietal lobe": ["PL.L", "PL.L_OL.L", "PL"],
#     "inferior parietal lobe": ["IPL-near-AG", "IPL.L"],
#
#     'occipital lobe': ['OL.L', 'OL.R'],
#
#
#     # 'posterior lateral ventricle': ['pLV.L'],
#     'others': ['CB_tonsil.L', 'FP.R', 'FP.L', 'BN.L', 'FT.L', 'CC.L', "TP", "TP.L", "TP.R"],
# }
# brain_region_abbreviation = {
#     'frontal lobe': 'FL',
#     "superior frontal gyrus": "SFG",
#     "middle frontal gyrus": "MFG",
#     "inferior frontal gyrus": "IFG",
#     #
#     'temporal lobe': 'TL',
#     "superior temporal gyrus": "STG",
#     "middle temporal gyrus": "MTG",
#     "parietal lobe": "PL",
#     "inferior parietal lobe": "IPL",
#     'occipital lobe': 'OL',
#     # 'posterior lateral ventricle': 'pLV',
#     'others': 'Others',
# }

brain_regions = {
    'Frontal': {
        "Frontal body": ["SFG.R", "SFG.L", "SFG"],  # 和superior frontal gyrus是同一个区域
        "Middle frontal": ["MFG.R", "MFG", "MFG.L"],
        "Inferior frontal": ["IFG", "IFG.R"],
        "Frontal pole": ['FP.L', 'FP.R'],
        # "Ambiguous (Frontal lobe)": ['FL.L', 'FL.R', '(X)FG', 'M(I)FG.L', 'S(M)FG.R', ],  # 后面两个是交叉脑区
    },
    'Parietal': {
        "Parietal": ["PL.L", "PL"],
        'Supramarginal': ["IPL", "IPL.L", 'IPL-near-AG'],  # 指的应该是同一个区域
    },
    'Temporal': {
        "Temporal body": ["STG.R", "STG", 'STG-AP', "S(M)TG.R", 'S(M)TG.L', "MTG.R", "MTG.L", "MTG"],
        "Temporal pole": ["TP.R", "TP", "TP.L"],
        # "Ambiguous (Temporal lobe)": ['TL.L', 'TL.R', 'S(M,I)TG', ]  # 后面三个是交叉脑区
    },
    'Occipital': {
        "Occipital": ['OL.L', 'OL.R']
    },
    "Ambiguous": {
        "Ambiguous": ["PL.L_OL.L", "FL_TL.L"]  # 不在allen的atlas里面
    }
}


brain_region_map = {}
for lobe in brain_regions:
    for region in brain_regions[lobe]:
        brain_region_map[region] = brain_regions[lobe][region]
print(brain_region_map)



neuron_info_file = r"/data/kfchen/trace_ws/paper_trace_result/csv_copy/Human_SingleCell_TrackingTable_20240712.csv"
neuron_info_df = pd.read_csv(neuron_info_file, encoding='gbk')
l_measure_files = [
    "/data2/kfchen/tracing_ws/14k_raw_img_data/lone_590_test_data_for_nnunet/8_estimated_radius_swc_l_measure.csv",
    "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc_l_measure.csv"
]
l_measure_df = pd.concat([pd.read_csv(f) for f in l_measure_files], ignore_index=True)

recon_list_files = ["train_val_list.csv", "test_list_with_gs.csv", "unlabeled_list.csv"]
total_neuron_id_list = []
for csv_file in recon_list_files:
    df = pd.read_csv(os.path.join("/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta", csv_file))
    current_id_list = df['id'].tolist()
    total_neuron_id_list.extend(current_id_list)
neuron_info_df = neuron_info_df[neuron_info_df['Cell ID'].isin(total_neuron_id_list)][['Cell ID', '脑区']]
# rename
neuron_info_df = neuron_info_df.rename(columns={'Cell ID': 'id', '脑区': 'brain_region'})
l_measure_df = l_measure_df[l_measure_df['ID'].isin(total_neuron_id_list)][['ID', 'Total Length', "Overall Width"]]
l_measure_df = l_measure_df.rename(columns={'ID': 'id', 'Total Length': 'total_length'})
neuron_info_df = neuron_info_df.merge(l_measure_df, on='id', how='left')









class swcPoint:
    def __init__(self, sample_number, structure_identifier,
                 x_position, y_position, z_position, radius, parent_sample):
        self.n = sample_number
        self.si = 0#structure_identifier
        self.x = x_position
        self.y = y_position
        self.z = z_position
        self.r = radius
        self.p = parent_sample
        self.s = [] # sons
        self.fn = -1 # fiber number
        self.conn = [] # connect points in other fiber
        self.mp = [] # match point in other swc
        self.neighbor = [] # neighbor closer than a distance. store neighbor number and connect info. as [d, bool]
        # self.isend = False
        self.ishead = False
        self.istail = False
        self.swcNeig = [] # neighbor closer than a distance.
        self.swcMatchP = []
        self.i = 0
        self.visited = 0
        self.pruned = False
        self.depth = 0




    def EndCheck(self):
        return self.ishead or self.istail


    def Printswc(self):
        print("n=%d, si=%d, x=%f, y=%f, z=%f, r=%f, p=%d, s=%s, fn=%d, neighbor=%s, mp=%s"
              %(self.n, self.si, self.x, self.y, self.z, self.r, self.p, str(self.s),
                self.fn, str(self.neighbor), str(self.mp)))

    def Writeswc(self, filepath, swcPoint_list,
                 reversal=False, limit=[256, 256, 128],
                 overlay=False, number_offset=0):
        if(reversal):
            line = "%d %d %f %f %f %f %d\n" %(
                self.n + number_offset, self.si, self.x,
                limit[1] - self.y,
                self.z, self.r, self.p + number_offset
            )
        else:
            line = "%d %d %f %f %f %f %d\n" %(
                self.n + number_offset, self.si, self.x,
                self.y,
                self.z, self.r, self.p + number_offset
            )
        if (overlay and os.path.exists(filepath)):
            # print("!!!!!!")
            os.remove(filepath)
        file_handle = open(filepath, mode="a")
        file_handle.writelines(line)
        file_handle.close()

class swcP_list:
    def __init__(self):
        self.p = []
        self.count = 0
def Readswc_v2(swc_name):
    point_l = swcP_list()
    with open(swc_name, 'r' ) as f:
        lines = f.readlines()

    swcPoint_number = -1
    # swcPoint_list = []
    point_list = []
    list_map = np.zeros(500000)

    for line in lines:
        if(line[0] == '#'):
            continue

        temp_line = line.split()
        # print(temp_line)
        point_list.append(temp_line)

        swcPoint_number = swcPoint_number + 1
        list_map[int(temp_line[0])] = swcPoint_number

    # print(point_list)
    swcPoint_number = 0
    for point in point_list:
        swcPoint_number = swcPoint_number + 1
        point[0] = swcPoint_number # int(point[0])
        point[1] = int(point[1])
        point[2] = float(point[2])
        point[3] = float(point[3])
        point[4] = float(point[4])
        point[5] = float(point[5])
        point[6] = int(point[6])
        if(point[6] == -1):
            pass
        else:
            point[6] = int(list_map[int(point[6])]) + 1

    # swcPoint_list.append(swcPoint(0,0,0,0,0,0,0)) # an empty point numbered 0
    point_l.p.append(swcPoint(0,0,0,0,0,0,0))

    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        point_l.p.append(temp_swcPoint)
    for point in point_list:
        temp_swcPoint = swcPoint(point[0], point[1], point[2], point[3], point[4], point[5], point[6])
        if not temp_swcPoint.p == -1:
            # parent = swcPoint_list[int(temp_swcPoint.p)]
            parent = point_l.p[int(temp_swcPoint.p)]
            parent.s.append(temp_swcPoint.n)
        if(point[0] == 1):
            point_l.p[int(point[0])].depth = 0
        else:
            point_l.p[int(point[0])].depth = parent.depth + 1
        # point_l.p.append(temp_swcPoint)
    # for i in range(1, 10):
    #     print(point_l.p[i].s)

    return point_l # (swcPoint_list)


def get_mip_swc(swc_file, projection_direction='xy', image=None, ignore_background=False,
                branch_color=(255, 0, 0), branch_thickness=2, soma_color=(255, 128, 0), soma_thickness=5,
                fig_size=(512, 512, 512), plot_center=True, border_color=(0, 0, 0), border_width_ratio=0.02):
    if projection_direction == 'xy':
        projection_axes = 0
    elif projection_direction == 'xz':
        projection_axes = 1
    elif projection_direction == 'yz':
        projection_axes = 2
    else:
        raise ValueError("Invalid projection direction. Choose from 'xy', 'xz', or 'yz'.")

    background = np.ones(fig_size, dtype=np.uint8) * 255
    background = np.max(background, axis=projection_axes)
    # print(background.shape)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

    point_l = Readswc_v2(swc_file)
    for p in point_l.p:
        if (p.n == 1):
            plot_offset = background.shape[0] / 2 - p.x, background.shape[1] / 2 - p.y, background.shape[2] / 2 - p.z
            break


    for p in point_l.p:
        if (p.n == 0 or p.n == 1): continue
        if (p.p == 0 or p.p == -1): continue
        x, y, z = p.x, p.y, p.z
        px, py, pz = point_l.p[p.p].x, point_l.p[p.p].y, point_l.p[p.p].z

        if(plot_center):
            x, y, z, px, py, pz = x + plot_offset[0], y + plot_offset[1], z + plot_offset[2], px + plot_offset[0], py + plot_offset[1], pz + plot_offset[2]

        x, y, z = int(x), int(y), int(z)
        px, py, pz = int(px), int(py), int(pz)

        if (projection_axes == 0):
            # draw a line between two poi
            cv2.line(background, (x, y), (px, py), branch_color, branch_thickness)
        elif (projection_axes == 1):
            cv2.line(background, (x, z), (px, pz), branch_color, branch_thickness)
        elif (projection_axes == 2):
            cv2.line(background, (y, z), (py, pz), branch_color, branch_thickness)

    soma_x, soma_y, soma_z = point_l.p[1].x, point_l.p[1].y, point_l.p[1].z
    if(plot_center):
        soma_x, soma_y, soma_z = soma_x + plot_offset[0], soma_y + plot_offset[1], soma_z + plot_offset[2]
    soma_x, soma_y, soma_z = int(soma_x), int(soma_y), int(soma_z)
    if (projection_axes == 0):
        cv2.circle(background, (soma_x, soma_y), soma_thickness, soma_color, -1)
    elif (projection_axes == 1):
        cv2.circle(background, (soma_x, soma_z), soma_thickness, soma_color, -1)
    elif (projection_axes == 2):
        cv2.circle(background, (soma_y, soma_z), soma_thickness, soma_color, -1)

    # if is float32
    if(isinstance(border_color[0], float)):
        # to RGB
        border_color = [int(c * 255) for c in border_color]
    border_width = int(border_width_ratio * np.min(fig_size))
    background = cv2.copyMakeBorder(background, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=[128, 128, 128])

    # 在左上角画一个圆
    cv2.circle(background, (30, 30), 10, border_color, -1)

    return background

def random_choose_sample(brain_region):
    neuron_ids = neuron_info_df[neuron_info_df['brain_region'].isin(brain_region_map[brain_region])]['id'].values
    neuron_id = np.random.choice(neuron_ids)
    return neuron_id

# 在中位数附近随机选择
def random_choose_sample_at_median(brain_region):
    filtered_df = neuron_info_df[neuron_info_df['brain_region'].isin(brain_region_map[brain_region])]
    total_length_median_75 = filtered_df['total_length'].quantile(0.9)
    total_length_median = total_length_median_75

    choose_range = (total_length_median * 0.9, total_length_median * 1.1)

    # filtered_df = filtered_df[(filtered_df['total_length'] >= choose_range[0]) & (filtered_df['total_length'] <= choose_range[1])]
    filtered_df = filtered_df[(filtered_df['total_length'] >= total_length_median)]


    neuron_ids = filtered_df['id'].values
    neuron_id = np.random.choice(neuron_ids)
    return neuron_id


def find_swc_file(neuron_id, swc_dir='/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc'):
    swc_files = [f for f in os.listdir(swc_dir) if int(f.split('_')[0]) == int(neuron_id)]
    return os.path.join(swc_dir, swc_files[0])

if __name__ == '__main__':
    insterested_brain_region = brain_region_map.keys()
    # colors = plt.cm.get_cmap('Set3').colors[:len(insterested_brain_region)]
    set2_colors = plt.cm.get_cmap('tab10').colors
    set2_colors = np.concatenate((set2_colors[:7], set2_colors[8:], [set2_colors[7]]), axis=0)
    colors = set2_colors

    # print(colors)
    row = 1
    col = (len(insterested_brain_region)) // row
    fig, axes = plt.subplots(row, col, figsize=(col * 2, row * 2))
    axex = axes.flatten()
    # 设置清晰度
    fig.set_dpi(300)
    for i, brain_region in enumerate(insterested_brain_region):
        neuron_id = random_choose_sample_at_median(brain_region)
        swc_file = find_swc_file(neuron_id)
        swc_mip = get_mip_swc(swc_file, border_color=colors[i], fig_size=(300, 300, 300), border_width_ratio=0.01)
        ax = axes[i]
        ax.imshow(swc_mip)
        ax.axis('off')
        # ax.set_title(f'No. {neuron_id} ({brain_region_abbreviation[brain_region]})')
        # to 00001
        neuron_id = str(neuron_id).zfill(5)
        # ax.text(0.5, 0.85, f'No. {neuron_id} ({brain_region_abbreviation[brain_region]})', ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.87, f'No. {neuron_id}', ha='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')



    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    # plt.show()
    plt.savefig('/data/kfchen/trace_ws/paper_trace_result/figs/1_trace_result_mip_v2.png')
    plt.close()

