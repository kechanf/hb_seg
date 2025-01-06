import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
                fig_size=(512, 512, 512), plot_center=True, border_color=(0, 0, 0), border_width_ratio=0.05):
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
    background = cv2.copyMakeBorder(background, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=border_color)

    return background


auto_swc_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc"
manual_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab"

df_auto = pd.read_csv(r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc_l_measure.csv")
df_manual = pd.read_csv(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv")

auto_ids = df_auto["ID"].values
manual_ids = df_manual["ID"].values
shared_ids = set(auto_ids) & set(manual_ids)

foucs_feature = [
    "Number of Branches",
    # 'Total Length',
]
# sample_threshold_max, sample_threshold_min, sample_threshold_step = 1.05, 0.8, 0.05
# sample_threshold_points = np.arange(sample_threshold_min, sample_threshold_max+sample_threshold_step, sample_threshold_step)
sample_threshold_points = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
print(sample_threshold_points)

chosen_sample = {
    "Number of Branches": {
        0.6: 2478,
        0.7: 3517,
        0.8: 2768,
        0.9: 2928,
        1.0: 3342,
        1.1: 2436,
    },
    "Total Length": {
    }
}

threshold = 0.8
for feature in foucs_feature:
    good_samples = []
    if(feature == "Number of Branches"):
        length_weighted_lm_file = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/comp_lm_df.csv"
        df_length_weighted_lm = pd.read_csv(length_weighted_lm_file)
        df_length_weighted_lm = df_length_weighted_lm[df_length_weighted_lm["id"].isin(shared_ids)]
        for id in shared_ids:
            length_weighted_value = df_length_weighted_lm[df_length_weighted_lm["id"] == id]["Length_Weighted_Number_of_Branches"].values[0]
            # if(auto_value > threshold * manual_value):
            good_samples.append((id, 0, 0, length_weighted_value))
    else:
        for id in shared_ids:
            auto_value = df_auto[df_auto["ID"] == id][feature].values[0]
            manual_value = df_manual[df_manual["ID"] == id][feature].values[0]
            # if(auto_value > threshold * manual_value):
            good_samples.append((id, auto_value, manual_value, auto_value/manual_value))
        # good_samples = sorted(good_samples, key=lambda x: x[3], reverse=False)

    random_samples = []

    for i in range(len(sample_threshold_points)-1):
        threshold1 = sample_threshold_points[i]
        threshold2 = sample_threshold_points[i+1]
        if(threshold1 in chosen_sample[feature]):
            random_samples.append((chosen_sample[feature][threshold1], 0, 0, 0))
            continue
        current_random_samples = []
        for sample in good_samples:
            if(threshold1 <= sample[3] < threshold2 or (i == 1 and sample[3] < threshold2)):
                current_random_samples.append(sample)
        print(f"Feature: {feature}, Threshold: {threshold1}-{threshold2}", len(current_random_samples))
        random_sample = random.choice(current_random_samples)
        random_samples.append(random_sample)

    print(f"Feature: {feature}", random_samples)

    row = 2
    col = len(sample_threshold_points)-1
    col = col
    print(row, col)
    fig, axs = plt.subplots(row, col, figsize=(col*4, row*4), dpi=300)
    axs = axs.flatten()

    # viridis
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(random_samples)))

    for i, sample in enumerate(random_samples):
        id = sample[0]
        auto_swc_path = os.path.join(auto_swc_dir, f"{id}.swc")
        manual_swc_path = os.path.join(manual_swc_dir, f"{id}.swc")

        swc_mip1 = get_mip_swc(auto_swc_path, fig_size=(300, 300, 300), border_width_ratio=0, branch_color=colors[i]*255)
        swc_mip2 = get_mip_swc(manual_swc_path, fig_size=(300, 300, 300), border_width_ratio=0, branch_color=colors[i]*255)

        axs[i].imshow(swc_mip1)
        axs[i+col].imshow(swc_mip2)
        # 左侧标题
        axs[i].set_title(f"r={sample_threshold_points[i]:.2f}", loc='left', fontsize=30, y=0.9)
        # x=0做一条线
        axs[i].axvline(x=0, color='gray', linestyle='--')
        axs[i+col].axvline(x=0, color='gray', linestyle='--')
        # axs[i*2+1].set_title(f"Manual: {id}")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig("/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/" + f"{feature}_sample.png")
    plt.close()






