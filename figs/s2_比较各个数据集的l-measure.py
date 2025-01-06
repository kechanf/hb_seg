import os
import glob
import numpy as np
import pandas as pd
from pylib.swc_handler import parse_swc, trim_swc, crop_spheric_from_soma
from simple_swc_tool.l_measure_api import l_measure_swc_dir
import shutil
import matplotlib.pyplot as plt
import cv2

feature_name_maps = {
    'Number of Branches': 'No. of Branches',
    'Total Length': 'Length (μm)',
    'Max Path Distance': 'Max Path Dist. (μm)',
    'N_stem': 'No. of Stems',
    'Number of Tips': 'No. of Tips',
    'Max Branch Order': 'Max Branch Order',
    'N_node': 'No. of Nodes',
    'Number of Bifurcatons': 'No. of Bifurcations',
    'Overall Width': 'Width (μm)',
    'Overall Height': 'Height (μm)',
    'Overall Depth': 'Depth (μm)',
    'Max Euclidean Distance': 'Max Euclidean (μm)',
    # 'Average Bifurcation Angle Remote': 'Avg. Remote BA (°)'
}

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

def crop_box_from_soma(swc_file, lim):
    x_lim, y_lim, z_lim = lim
    df = pd.read_csv(swc_file, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
    soma = df[df.type == 1 & (df.pid == -1)]
    assert len(soma) == 1
    soma = soma.iloc[0]
    soma_x, soma_y, soma_z = soma[['x', 'y', 'z']]

    x_min, x_max = soma_x - x_lim, soma_x + x_lim
    y_min, y_max = soma_y - y_lim, soma_y + y_lim
    z_min, z_max = soma_z - z_lim, soma_z + z_lim
    df_crop = df[(df.x >= x_min) & (df.x <= x_max) &
                 (df.y >= y_min) & (df.y <= y_max) &
                 (df.z >= z_min) & (df.z <= z_max)]
    return df_crop

def get_swc_dims(swcfile):
    df = pd.read_csv(swcfile, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
    xmin, ymin, zmin = np.min(df[['x', 'y', 'z']], axis=0)
    xmax, ymax, zmax = np.max(df[['x', 'y', 'z']], axis=0)
    return xmin, ymin, zmin, xmax, ymax, zmax

def shperic_cropping(in_dir, out_dir, radius, remove_axon=True):
    ninswc = 0
    for inswc in glob.glob(os.path.join(in_dir, '*.swc')):
        filename = os.path.split(inswc)[-1]
        outswc = os.path.join(out_dir, filename)
        if os.path.exists(outswc):
            continue
        ninswc += 1
        # if ninswc % 20 == 0:
        #     print(filename)
        df_tree = pd.read_csv(inswc, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
        if remove_axon:
            df_tree = df_tree[df_tree.type != 2]

        # cropping
        tree_out = crop_spheric_from_soma(df_tree, radius)

        tree_out.to_csv(outswc, sep=' ', index=True, header=False)

def soma_converter(swcfile, out_dir):
    outfile = os.path.join(out_dir, os.path.split(swcfile)[-1])
    if os.path.exists(outfile):
        return

    try:
        df = pd.read_csv(swcfile, comment='#', sep=' ', usecols=range(1, 8),
                         names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'), index_col=0)
    except pd.errors.ParserError:
        df = pd.read_csv(swcfile, comment='#', sep=' ', usecols=range(0, 7),
                         names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'), index_col=0)

    # find out the non-center soma point
    ncs = (df.type == 1) & (df.pid != -1)
    # print(swcfile, ncs.sum())
    if ncs.sum() == 1:
        # the SWC is fragmentated in this case!
        idx = df.index[np.nonzero(ncs)[0]][0]
        df = df.iloc[np.nonzero(~ncs)[0]]

        assert (df.pid == idx).sum() == 0
        indices = df.index.values.copy()
        indices[indices > idx] = indices[indices > idx] - 1
        df.index = indices

        pindices = df.pid.values.copy()
        pindices[pindices > idx] = pindices[pindices > idx] - 1
        df.pid = pindices
    elif ncs.sum() == 2:
        # re-ordering all nodes
        id1, id2 = df.index[np.nonzero(ncs)[0]]
        # make sure no nodes are connected to these two points
        assert (df.pid == id1).sum() == 0
        assert (df.pid == id2).sum() == 0
        if id1 > id2:
            id1, id2 = id2, id1

        df = df.iloc[np.nonzero(~ncs)[0]]
        indices = df.index.values.copy()
        indices[indices > id2] = indices[indices > id2] - 1
        indices[indices > id1] = indices[indices > id1] - 1
        df.index = indices

        # also for the parent indices
        pindices = df.pid.values.copy()
        pindices[pindices > id2] = pindices[pindices > id2] - 1
        pindices[pindices > id1] = pindices[pindices > id1] - 1
        df.pid = pindices
    elif ncs.sum() > 2:
        raise ValueError

    # save out
    df.to_csv(outfile, sep=' ', index=True, header=False)

def move_proposed():
    source_swc_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc"
    target_swc_dir = "/data/kfchen/trace_ws/quality_control_test/proposed/CNG_version"

    unlabeled_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv"
    test_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
    unlabeled_list = pd.read_csv(unlabeled_list_file)["id"].tolist()
    test_list = pd.read_csv(test_list_file)["id"].tolist()

    for infile in glob.glob(os.path.join(source_swc_dir, '*.swc')):
        id = int(os.path.basename(infile).split("_")[0])
        if id in unlabeled_list or id in test_list:
            outfile = os.path.join(target_swc_dir, os.path.basename(infile))
            if(not os.path.exists(outfile)):
                shutil.copy(infile, outfile)



# move_proposed()
source_swc_root = "/home/sujun/lyf/public_data"
dataset_name = [
    "allen_human_neuromorpho",
    "allman",
    "ataman_boulting",
    "DeKock",
    "hrvoj-mihic_semendeferi",
    "jacobs",
    "segev",
    "semendeferi_muotri",
    "vdheuvel",
    "vuksic",
    "wittner",
    "proposed",
]
instersted_dataset_name = [
    "allen_human_neuromorpho",
    "allman",
    # "ataman_boulting",
    "DeKock",
    "hrvoj-mihic_semendeferi",
    "jacobs",
    # "segev",
    "semendeferi_muotri",
    "vdheuvel",
    # "vuksic",
    # "wittner",
    "proposed",
]

dataset_name_map = {
    "allen_human_neuromorpho": "Allen",
    "allman": "Allman",
    "DeKock": "DeKock",
    "hrvoj-mihic_semendeferi": "Hrvoj",
    "jacobs": "Jacobs",
    "semendeferi_muotri": "Semendeferi",
    "vdheuvel": "vdHeuvel",
    "proposed": "Proposed",
}

target_swc_root = "/data/kfchen/trace_ws/quality_control_test/human"

def main(flag="spheric", radius=None, lim=None):
    assert flag in ["spheric", "box"]
    for name in instersted_dataset_name:
        if(name == "proposed"):
            origin_swc_dir = "/data/kfchen/trace_ws/quality_control_test/proposed/raw"
        else:
            origin_swc_dir = os.path.join(source_swc_root, name, "raw")
        one_soma_swc_dir = os.path.join(target_swc_root, name, "one_point_soma")

        # os.makedirs(one_soma_swc_dir, exist_ok=True)
        # for infile in glob.glob(os.path.join(origin_swc_dir, '*.swc')):
        #     # soma_converter(infile, one_soma_swc_dir)
        #     try:
        #         soma_converter(infile, one_soma_swc_dir)
        #     except:
        #         pass

        # if flag == "spheric":
        #     one_soma_cropped_dir = os.path.join(target_swc_root, name, "cropped_" + str(radius) + "um")
        #     if(not os.path.exists(one_soma_cropped_dir)):
        #         os.makedirs(one_soma_cropped_dir, exist_ok=True)
        #         shperic_cropping(one_soma_swc_dir, one_soma_cropped_dir, radius)
        # elif flag == "box":
        #     one_soma_cropped_dir = os.path.join(target_swc_root, name, "one_point_soma_box_" + str(lim[0]) + str(lim[1]) + "um")
        #     if(not os.path.exists(one_soma_cropped_dir)):
        #         os.makedirs(one_soma_cropped_dir, exist_ok=True)
        #         for infile in glob.glob(os.path.join(one_soma_swc_dir, '*.swc')):
        #             try:
        #                 df = crop_box_from_soma(infile, lim)
        #                 outfile = os.path.join(one_soma_cropped_dir, os.path.basename(infile))
        #                 df.to_csv(outfile, sep=' ', index=True, header=False)
        #             except:
        #                 pass

        if(flag == "spheric"):
            l_measure_result_file = os.path.join(target_swc_root, name, "l_measure_" + str(radius) + "um.csv")
        elif(flag == "box"):
            l_measure_result_file = os.path.join(target_swc_root, name, "one_point_soma_box_" + str(lim[0]) + str(lim[1]) + "um_l_measure.csv")
        print(l_measure_result_file)
        # if(not os.path.exists(l_measure_result_file)):
        #     l_measure_swc_dir(one_soma_cropped_dir, l_measure_result_file)
        print(name, radius, "done")

    l_measure_result_files = []
    for name in instersted_dataset_name:
        if(flag == "spheric"):
            l_measure_result_files.append(os.path.join(target_swc_root, name, "l_measure_" + str(radius) + "um.csv"))
            print(l_measure_result_files[-1])
        elif(flag == "box"):
            l_measure_result_files.append(os.path.join(target_swc_root, name, "one_point_soma_box_" + str(lim[0]) + str(lim[1]) + "um_l_measure.csv"))
    dfs = [pd.read_csv(f) for f in l_measure_result_files]
    proposed_df = pd.read_csv(l_measure_result_files[-1])
    full_recon_list = pd.read_csv("/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_recon_list.csv")["id"].tolist()
    proposed_df = proposed_df[proposed_df["ID"].isin(full_recon_list)]
    print(len(proposed_df))
    dfs = dfs[:-1]
    dfs.append(proposed_df)


    # print(len(dfs))
    instersted_feasures = [
        'N_stem', 'Number of Bifurcatons',
        'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
        'Overall Depth', 'Total Length',
        'Max Euclidean Distance', 'Max Path Distance',
        'Max Branch Order',
    ]
    # 设置分辨率
    # plt.rcParams['figure.dpi'] = 800
    col = 3
    row = 4
    fig, axs = plt.subplots(row, col, figsize=(col * 5, row * 5))
    axs = axs.flatten()

    colors = plt.get_cmap('Set3').colors
    for feasure, ax in zip(instersted_feasures, axs):
        # ax.set_title(feasure)
        ax.set_ylabel(feature_name_maps[feasure])
        positions = range(len(instersted_dataset_name))

        for i, df in enumerate(dfs):
            # print(i, len(df[feasure]))
            ax.boxplot(df[feasure].dropna(), positions=[i], widths=0.5, patch_artist=True,
                           showfliers=True, boxprops=dict(facecolor=colors[i], color='black'),
                           medianprops=dict(color='black'), flierprops=dict(marker='o', color='black', markersize=3))
            # print(df[feasure])
        ax.set_xticks(positions)
        xticklabels = [dataset_name_map[name] for name in instersted_dataset_name]
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')

    # 关闭空白子图
    for i in range(len(instersted_feasures), col * row):
        fig.delaxes(axs[i])

    plt.tight_layout()
    if(flag == "spheric"):
        plt_file = "/data/kfchen/trace_ws/quality_control_test/"+str(radius)+"um_l_measure.png"
    elif(flag == "box"):
        plt_file = "/data/kfchen/trace_ws/quality_control_test/box_"+ str(lim[0]) + str(lim[1]) +"um_l_measure.png"
    plt.savefig(plt_file)
    print(plt_file)
    plt.close()

    # random sample
    instersted_swc_dir = []
    for name in instersted_dataset_name:
        if(flag == "spheric"):
            instersted_swc_dir.append(os.path.join(target_swc_root, name, "one_point_soma_" + str(radius) + "um"))
        elif(flag == "box"):
            instersted_swc_dir.append(os.path.join(target_swc_root, name, "one_point_soma_box_" + str(lim[0]) + str(lim[1]) + "um"))
    # sample_num = 5
    # col = sample_num
    # row = len(instersted_dataset_name)
    # fig, axs = plt.subplots(row, col, figsize=(col * 5, row * 5))
    # axs = axs.flatten()
    #
    # for i, swc_dir in enumerate(instersted_swc_dir):
    #     for j, swc_file in enumerate(np.random.choice(glob.glob(os.path.join(swc_dir, '*.swc')), sample_num)):
    #         ax = axs[i * sample_num + j]
    #         if(flag == "spheric"):
    #             fig_size = (radius * 2, radius * 2, radius * 2)
    #         elif(flag == "box"):
    #             fig_size = lim
    #         ax.imshow(get_mip_swc(swc_file, projection_direction='xy', fig_size=fig_size, branch_thickness=1, border_width_ratio=0.01))
    #         ax.axis('off')
    #         ax.set_title(os.path.basename(swc_file[:5]))
    # plt.tight_layout()
    # if(flag == "spheric"):
    #     plt_file = "/data/kfchen/trace_ws/quality_control_test/"+str(radius)+"um_sample.png"
    # elif(flag == "box"):
    #     plt_file = "/data/kfchen/trace_ws/quality_control_test/box_"+ str(lim[0]) + str(lim[1]) +"um_sample.png"
    # plt.savefig(plt_file)
    # plt.close()




# main("spheric", 50)
# main("spheric", 100)
main("spheric", 150)
#
# main("box", None, (50, 50, 50))
# main("box", None, (100, 100, 50))
# main("box", None, (150, 150, 50))
