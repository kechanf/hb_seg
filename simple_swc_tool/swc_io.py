import math
import time

import numpy as np

import queue
import seaborn as sns
from simple_swc_tool.swc_base import *

def read_swc(swc_name):
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

def write_swc(filepath, point_l, fiber_l = None, reversal=False, limit=[1000, 1000, 1000], overlay=False, number_offset=0):
    lines = []
    for temp_p in point_l.p:
        if(temp_p.n == 0):continue
        if(fiber_l):
            if(fiber_l.f[temp_p.fn - 1].pruned):continue
        if(temp_p.pruned):continue

        # if(temp_p.n not in fiber_l.f[temp_p.fn - 1].p):continue
        # if(temp_p.ishead): continue
        # print(temp_p.n)
        if (reversal):
            line = "%d %d %f %f %f %f %d\n" % (
                temp_p.n + number_offset, temp_p.si, temp_p.x,
                limit[1] - temp_p.y,
                temp_p.z, temp_p.r, temp_p.p + number_offset
            )
        else:
            line = "%d %d %f %f %f %f %d\n" % (
                temp_p.n + number_offset, temp_p.si, temp_p.x,
                temp_p.y,
                temp_p.z, temp_p.r, temp_p.p + number_offset
            )
        lines.append(line)

    if (overlay and os.path.exists(filepath)):
        # print("!!!!!!")
        os.remove(filepath)
    file_handle = open(filepath, mode="a")
    file_handle.writelines(lines)
    file_handle.close()