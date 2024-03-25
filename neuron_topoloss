import os.path

import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import gudhi as gd
import time

import cProfile
import pstats
# import torch
from concurrent.futures import ThreadPoolExecutor
import tifffile

# import dionysus as d
# from ripser import ripser


def get_blocks(gt, pred, block_size):
    # print(f"gt.shape: {gt.shape}, pred.shape: {pred.shape}")
    num_blocks = [np.ceil(gt.shape[dim] / block_size[dim]).astype(int) for dim in range(3)]
    # print(f"num_blocks: {num_blocks}")

    # 初始化用于存储切块的列表
    gt_blocks = []
    pred_blocks = []

    # 进行切块
    for z in range(num_blocks[0]):
        for y in range(num_blocks[1]):
            for x in range(num_blocks[2]):
                z_start, z_end = z * block_size[0], min((z + 1) * block_size[0], gt.shape[0])
                y_start, y_end = y * block_size[1], min((y + 1) * block_size[1], gt.shape[1])
                x_start, x_end = x * block_size[2], min((x + 1) * block_size[2], gt.shape[2])

                gt_block = gt[z_start:z_end, y_start:y_end, x_start:x_end]
                pred_block = pred[z_start:z_end, y_start:y_end, x_start:x_end]

                if(gt_block.sum() or pred_block.sum()):
                    gt_blocks.append(gt_block)
                    pred_blocks.append(pred_block)

    return gt_blocks, pred_blocks

def gd_ph(image): # 调库
    image = 1 - image
    # 将图像转换为立方复合体（CubicalComplex）
    cc = gd.CubicalComplex(top_dimensional_cells=image.flatten(), dimensions=image.shape)
    # 计算持续同调
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    # # 获取0维的持续同调特征
    per0 = cc.persistence_intervals_in_dimension(0)
    for i in range(len(per0)):
        if(per0[i][1] == np.inf):
            per0[i][1] = 1
    return per0

def process_block(block_pair): # 单线程20.740 # 多线程1.320s
    gt_block, pred_block = block_pair # my 16s # gd 0.4s
    gt_per = []
    pred_per = []
    pred_per = gd_ph(pred_block)
    gt_per = gd_ph(gt_block)
    return get_topoloss(gt_per, pred_per)

def block_test():
    image = 1 - pred_blocks[15]
    cc = gd.CubicalComplex(top_dimensional_cells=image.flatten(), dimensions=image.shape)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    pred_per0 = cc.persistence_intervals_in_dimension(0)
    gd.plot_persistence_barcode(pred_per0)
    plt.title('0-Dimensional Persistence Barcode')
    num1 = len(cc.persistence_intervals_in_dimension(1))

    image = 1 - gt_blocks[15]
    cc = gd.CubicalComplex(top_dimensional_cells=image.flatten(), dimensions=image.shape)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    gt_per0 = cc.persistence_intervals_in_dimension(0)
    gd.plot_persistence_barcode(gt_per0)
    plt.title('0-Dimensional Persistence Barcode')
    num2 = len(cc.persistence_intervals_in_dimension(1))
    print(f"len of gt: {num2}, and len of pred: {num1}")

    plt.show()
    topoloss(gt_per0, pred_per0)

    label_image = label(image, structure=np.ones((3, 3, 3)))[0]
    # print number
    print(f"number of regions is {np.max(label_image)}")

    tifffile.imwrite(r'gt_block.tif', (gt_blocks[15] * 255).astype(np.uint8))
    tifffile.imwrite(r'pred_block.tif', (pred_blocks[15] * 255).astype(np.uint8))

from gudhi.hera import wasserstein_distance

def get_topoloss(gt_per0, pred_per0):
    bottleneck_distance = gd.bottleneck_distance(gt_per0, pred_per0)
    # print(f"bottleneck_distance: {bottleneck_distance}")
    return bottleneck_distance




def toy_test():
    # toy_path = r"C:\Users\12626\Desktop\topo_test\toy.tif"
    toy_size = (10, 20, 20)
    toy = np.zeros(toy_size, dtype=np.float64)
    # # create_sphere(toy, center=(5, 5, 5), radius=1)
    # # create_sphere(toy, center=(15, 15, 15), radius=1)
    #
    #

    toy[5:7, 5:7, 1:3] = 1
    toy[5:7, 5:7, 3:5] = 1
    toy[5:7, 5:7, 5:7] = 1

    cc = gd.CubicalComplex(top_dimensional_cells=(1-toy).flatten(), dimensions=toy.shape)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    per0 = cc.persistence_intervals_in_dimension(0)
    for i in range(len(per0)):
        if(per0[i][1] == np.inf):
            per0[i][1] = 1
    gd.plot_persistence_barcode(per0)
    plt.title('0-Dimensional Persistence Barcode')
    pred_per0 = per0


    toy[5:7, 5:7, 1:3] = 0
    toy[5:7, 5:7, 3:5] = 0
    toy[5:7, 5:7, 5:7] = 0

    cc = gd.CubicalComplex(top_dimensional_cells=(1-toy).flatten(), dimensions=toy.shape)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    per0 = cc.persistence_intervals_in_dimension(0)
    for i in range(len(per0)):
        if(per0[i][1] == np.inf):
            per0[i][1] = 1
    gd.plot_persistence_barcode(per0)
    plt.title('0-Dimensional Persistence Barcode')
    gt_per0 = per0

    plt.show()

    print(topoloss(gt_per0, pred_per0))

    #
    # # toy = 1 - toy
    # # toy[1:2, 1:2, 1:2] = 1
    #
    # # toy_size = (10, 10)
    # # toy = np.random.rand(*toy_size)
    # # toy = np.zeros(toy_size, dtype=np.uint8)
    # # toy[5:7, 5:7] = 1
    # # toy[2:4, 2:4] = 1
    #
    #
    #
    # print(f"toy.sum(): {toy.sum()}")
    # sigmoid

    # toy_flatten = toy.flatten()
    # for i in range(len(toy_flatten)):
    #     if(toy_flatten[i] > 0.1):
    #         print(i, toy_flatten[i])

    # toy = 1 / (1 + np.exp(-toy))



    # # tifffile.imwrite(toy_path, toy)
    # # cc = gd.CubicalComplex(top_dimensional_cells=toy.flatten(), dimensions=[20, 20, 10])
    # pers = cc.persistence(min_persistence=0, homology_coeff_field=2)
    # res = np.array([list(b) for (_, b) in pers])
    # print(f"res: {res}")
    # print(cc.dimension())
    # print(cc.cofaces_of_persistence_pairs()[0])
    # print(cc.persistence_intervals_in_dimension(2))
    # print(len(cc.persistence_intervals_in_dimension(2)))
    # gd.plot_persistence_barcode(cc.cofaces_of_persistence_pairs())
    # plt.title('0-Dimensional Persistence Barcode')
    # plt.show()
    pass

# toy_test()
def topoloss(gt, pred, block_size=(10,20,20), cProfile_on=False):
    if(cProfile_on):
        pr = cProfile.Profile()
        pr.enable()

    gt_blocks, pred_blocks = get_blocks(gt, pred, block_size)

    # loss = 0
    # for i in range(len(gt_blocks)):
    #     loss = loss + process_block((gt_blocks[i], pred_blocks[i]))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_block, zip(gt_blocks, pred_blocks)))
    loss = sum(results)
    #
    # if(cProfile_on):
    #     pr.disable()
    #     ps = pstats.Stats(pr).sort_stats('cumulative')
    #     ps.print_stats()
    return loss

if __name__ == '__main__':
    gt = tifffile.imread(r'C:\Users\12626\Desktop\topo_test\y4.tif')
    # pred = np.zeros_like(gt)
    pred = np.random.rand(*gt.shape) * 255
    # pred = np.ones_like(gt) * 255
    # pred = tifffile.imread(r'C:\Users\12626\Desktop\topo_test\x4.tif')
    gt = gt[0,0,:].astype(np.uint8) / 255
    pred = pred[0,0,:].astype(np.float64) / 255

    loss = topoloss(gt, pred, cProfile_on=False)

    print(loss)






