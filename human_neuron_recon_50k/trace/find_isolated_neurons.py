import os
import pandas as pd
from tqdm import tqdm

img_size = [100, 700, 700]
# 假设neuron到最远的branch的距离是150体素，则要求neuron之间至少间隔300体素
# 到最近的neuron的距离大于700体素的样本数量: 707 # 几乎不会受到其他神经元分支的影响
# 到最近的neuron的距离大于500体素的样本数量: 1889 # 偶尔会有一点影响
# 到最近的neuron的距离大于300体素的样本数量: 7216 # 可能会受到影响
# 到最近的neuron的距离大于100体素的样本数量: 38934 # 容易受到影响

marker_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/PTRSB_Somas"
marker_files = [f for f in os.listdir(marker_dir) if f.endswith('.marker')]
total_points = []
isolated_points = []
for marker_file in tqdm(marker_files):
    marker_file_path = os.path.join(marker_dir, marker_file)
    ptrs_markers = pd.read_csv(marker_file_path, sep=',',
                               comment='#',
                               names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
                                      'color_b'])

    points = []
    for i in range(len(ptrs_markers)):
        name, x, y, z = ptrs_markers.iloc[i][['name', 'x', 'y', 'z']]
        x, y, z = float(x), float(y), float(z)
        points.append((name, x, y, z))
    total_points.extend(points)
    for i in points:
        flag = True
        for j in points:
            if(i == j):
                continue
            # print(i[1] - j[1], i[2] - j[2])
            if (i[1] - j[1])**2 + (i[2] - j[2])**2 < img_size[0] **2:
                flag = False
                break
        if flag:
            isolated_points.append(i[0])

print(len(total_points))
print(len(isolated_points))


