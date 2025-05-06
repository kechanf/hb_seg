import os
import tifffile

from nnUNet.nnunetv2.dataset_conversion.generate_nnunet_dataset import augment_gamma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from nnUNet.scripts.mip import get_mip_swc

neuron_info_df = pd.read_csv("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv", encoding='gbk')

def find_resolution(filename):
    # print(filename)
    df = neuron_info_df
    filename = int(filename.split('.')[0].split('_')[0])
    for i in range(len(df)):
        if int(df.iloc[i, 0]) == filename:
            return df.iloc[i, 43]

def find_swc(swc_dir, tif_file):
    file_name = tif_file[:-4]
    swc_files = os.listdir(swc_dir)
    for swc_file in swc_files:
        if int(file_name.split("_")[0]) == int(swc_file.split("_")[0].split(".")[0]):
            return os.path.join(swc_dir, swc_file)


def plot_compare_result(file_list, img_dir, seg_dirs, seg_labels, recon_dirs, recon_labels, info_file, index):
    row = 2
    col = 6
    plt.figure(figsize=(5 * col, row * 5), dpi=800)

    for idx, img_file in enumerate(file_list):
        img_path = os.path.join(img_dir, img_file)
        seg_paths = [os.path.join(seg_dir, img_file) for seg_dir in seg_dirs]
        recon_paths = [find_swc(recon_dir, img_file) for recon_dir in recon_dirs]
        xy_resolution = find_resolution(os.path.basename(recon_paths[0]))

        img = tifffile.imread(img_path)
        img = augment_gamma(img)
        # img = resize(img, (img.shape[0], int(img.shape[1] * xy_resolution / 1000), int(img.shape[2] * xy_resolution / 1000), 1), order=2, mode='constant', cval=0, clip=True, preserve_range=True)
        img_mip = np.max(img, axis=0)


        segs = [tifffile.imread(seg_path) for seg_path in seg_paths]
        # segs[0] = skimage.transform.resize(segs[0], (img.shape[1], img.shape[2], img.shape[0]), order=0, mode='constant', cval=0, clip=True, preserve_range=True)
        segs = [(seg - np.min(seg)) / (np.max(seg) - np.min(seg)) * 255 for seg in segs]
        seg_mips = [np.max(seg, axis=0) for seg in segs]
        # seg_mips = [resize(seg_mip, (int(seg_mip.shape[0] * xy_resolution / 1000), int(seg_mip.shape[1] * xy_resolution / 1000)), order=0, mode='constant', cval=0, clip=True, preserve_range=True) for seg_mip in seg_mips]



        recon_mips = [get_mip_swc(recon_path, img, ignore_background=True) for recon_path in recon_paths]
        recon_mips = [recon_mip[:int(recon_mip.shape[0] * xy_resolution / 1000), :int(recon_mip.shape[1] * xy_resolution / 1000)] for recon_mip in recon_mips]

        info_df = pd.read_csv(info_file, encoding='gbk')
        ID = int(img_file.split(".")[0])
        brain_region = info_df[info_df['Cell ID'] == ID]['脑区'].values[0]
        label_info = f"No.0{img_file.split('.')[0]} ({brain_region})"

        default_size = int(512 * xy_resolution / 1000)
        img_mip = cv2.resize(img_mip, (default_size, default_size))
        seg_mips = [cv2.resize(seg_mip, (default_size, default_size)) for seg_mip in seg_mips]

        img_mip = cv2.cvtColor(img_mip.astype('uint8'), cv2.COLOR_GRAY2RGB)
        seg_mips = [cv2.cvtColor(seg_mip.astype('uint8'), cv2.COLOR_GRAY2RGB) for seg_mip in seg_mips]

        recon_mips = [cv2.resize(recon_mip, (default_size, default_size)) for recon_mip in recon_mips]
        # recon_mips = [cv2.rectangle(recon_mip, (0, 0), (recon_mip.shape[1], recon_mip.shape[0]), (0, 0, 0), 3) for
        #               recon_mip in recon_mips] # 添加边框

        labels = [label_info, seg_labels[0], recon_labels[0], seg_labels[1], recon_labels[1], seg_labels[2], recon_labels[2]]
        plot_mips = [img_mip, seg_mips[0],  seg_mips[1], seg_mips[2], seg_mips[3], seg_mips[4],
                     recon_mips[0], recon_mips[1], recon_mips[2], recon_mips[3], recon_mips[4]]

        crop_size = 200
        plot_mips = [plot_mip[default_size//2-crop_size//2:default_size//2+crop_size//2, default_size//2-crop_size//2:default_size//2+crop_size//2] for plot_mip in plot_mips]
        plot_mips[6:] = [cv2.rectangle(plot_mip, (0, 0), (plot_mip.shape[1], plot_mip.shape[0]), (0, 0, 0), 3) for plot_mip in plot_mips[6:]]


        for i, plot_mip in enumerate(plot_mips):
            # print(plot_mip.shape)
            flag = 1 if i >= 6 else 0
            # 横向
            ax = plt.subplot(row, col, idx * col + i + 1 + flag)
            # 纵向
            # ax = plt.subplot(7, 5, i * 5 + idx + 1)
            ax.imshow(plot_mip)
            ax.axis('off')
            # if(i == 0):
            #     ax.text(plot_mip.shape[1] / 2, 20, labels[i], fontsize=30, color='white', ha='center', va='top',
            #             backgroundcolor='black')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.savefig(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/compare_result/" + str(index) + ".png")
    # plt.show()
    plt.savefig(r"/data/kfchen/trace_ws/paper_trace_result/nnunet/temp_mip/compare_result" + str(index) + ".png")
    plt.close()

def plot_compare_result_0(file_list, img_dir, seg_dirs, seg_labels, info_file, index):
    plt.figure(figsize=(5 * 5, 4 * 5))

    for idx, img_file in enumerate(file_list):
        img_path = os.path.join(img_dir, img_file)
        seg_paths = [os.path.join(seg_dir, img_file) for seg_dir in seg_dirs]

        img = tifffile.imread(img_path)
        img = augment_gamma(img)
        img_mip = np.max(img, axis=0)

        segs = [tifffile.imread(seg_path) for seg_path in seg_paths]
        segs = [(seg - np.min(seg)) / (np.max(seg) - np.min(seg)) * 255 for seg in segs]
        seg_mips = [np.max(seg, axis=0) for seg in segs]

        info_df = pd.read_csv(info_file, encoding='gbk')
        ID = int(img_file.split(".")[0])
        brain_region = info_df[info_df['Cell ID'] == ID]['脑区'].values[0]
        label_info = f"No.0{img_file.split('.')[0]} ({brain_region})"

        default_size = 256
        img_mip = cv2.resize(img_mip, (default_size, default_size))
        seg_mips = [cv2.resize(seg_mip, (default_size, default_size)) for seg_mip in seg_mips]

        img_mip = cv2.cvtColor(img_mip.astype('uint8'), cv2.COLOR_GRAY2RGB)
        seg_mips = [cv2.cvtColor(seg_mip.astype('uint8'), cv2.COLOR_GRAY2RGB) for seg_mip in seg_mips]

        ax1 = plt.subplot(4, 5, idx + 1)
        ax1.imshow(img_mip)
        ax1.axis('off')
        ax1.text(img_mip.shape[1] / 2, 20, label_info, fontsize=30, color='white', ha='center', va='top',
                 backgroundcolor='black')

        for i, seg_mip in enumerate(seg_mips):
            ax = plt.subplot(4, 5, 5 + idx + 1 + i * 5)
            ax.imshow(seg_mip)
            ax.axis('off')
            # ax.text(seg_mip.shape[1] / 2, 20, seg_labels[i], fontsize=30, color='white', ha='center', va='top',
            #         backgroundcolor='black')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.savefig(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/compare_result/" + str(index) + ".png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    img_dir = r"/data/kfchen/trace_ws/to_gu/img"
    info_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
    seg_dirs = [
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/label",
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/baseline/0_seg",
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/cldice/0_seg",
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/skelrec/0_seg",
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/0_seg",
    ]
    seg_labels = [
        "Label",
        "Baseline",
        "Proposed",
    ]
    recon_dirs = [
        r'/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab',
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/baseline/8_estimated_radius_swc",
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/cldice/8_estimated_radius_swc",
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/skelrec/8_estimated_radius_swc",
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc",
    ]
    recon_labels = [
        "Manual Reconstruction",
        "Baseline Reconstruction",
        "Proposed Reconstruction",
    ]

    # tif
    img_files = os.listdir(seg_dirs[-1])
    img_files = [img_file for img_file in img_files if img_file.endswith(".tif")]
    # img_files = random.sample(img_files, 2)
    img_files = ["2882.tif", "2894.tif", "3507.tif", "3212.tif", "2861.tif"]
    # plot_compare_result_0(img_files, img_dir, seg_dirs, seg_labels, info_file, "chosen")
    # plot_compare_result(img_files, img_dir, seg_dirs, seg_labels, recon_dirs, recon_labels, info_file, "chosen")
    for img_file in img_files:
        plot_compare_result([img_file], img_dir, seg_dirs, seg_labels, recon_dirs, recon_labels, info_file, img_file.split(".")[0])

    # for index in range(int(len(img_files)/5)):
    #     plot_compare_result(img_files[index*5:index*5+5], img_dir, seg_dirs, seg_labels, info_file, index)

'''
2882
2894

3507
3212
2861

'''


