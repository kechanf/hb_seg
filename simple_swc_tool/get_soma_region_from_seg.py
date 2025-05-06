import os
import shutil
import subprocess
import time
import uuid
from functools import partial
from multiprocessing import Pool

import cc3d
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage.morphology
import tifffile
from scipy.ndimage import binary_dilation
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops, label
from skimage.morphology import ball
from skimage.morphology import skeletonize_3d
from tqdm import tqdm

# os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"


import cupy as cp
from cupyx.scipy.ndimage import binary_opening
import cupyx

import SimpleITK as sitk
# from gcut.python.neuron_segmentation import NeuronSegmentation
import sys
from simple_swc_tool.swc_io import read_swc, write_swc

from scipy import ndimage
# from nnunetv2.training.loss.fastanison import anisodiff3
import pandas as pd
from pylib.file_io import load_image
import numpy as np
from scipy.ndimage import label


v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"

class SomaRegionFinder():
    def process_path(self, pstr):
        return pstr.replace('(', '\(').replace(')', '\)')

    def dusting(self, img):
        if (img.sum() == 0):
            return img
        labeled_image = cc3d.connected_components(img, connectivity=6)
        largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
        largest_component_binary = ((labeled_image == largest_label)).astype("uint8")
        return largest_component_binary

    def crop_nonzero(self, image):
        non_zero_coords = np.argwhere(image)
        min_coords = non_zero_coords.min(axis=0)
        max_coords = non_zero_coords.max(axis=0) + 1

        cropped_image = image[min_coords[0]:max_coords[0],
                        min_coords[1]:max_coords[1],
                        min_coords[2]:max_coords[2]]

        return cropped_image, image.shape, min_coords

    def restore_original_size(self, cropped_image, original_shape, min_coords):
        restored_image = np.zeros(original_shape, dtype=cropped_image.dtype)
        restored_image[min_coords[0]:min_coords[0] + cropped_image.shape[0],
        min_coords[1]:min_coords[1] + cropped_image.shape[1],
        min_coords[2]:min_coords[2] + cropped_image.shape[2]] = cropped_image

        return restored_image

    def get_min_diameter_3d(self, binary_image):
        labeled_array, num_features = scipy.ndimage.label(binary_image)
        largest_cc = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
        slice_x, slice_y, slice_z = find_objects(labeled_array == largest_cc)[0]
        diameter_x = slice_x.stop - slice_x.start
        diameter_y = slice_y.stop - slice_y.start
        diameter_z = slice_z.stop - slice_z.start

        return min(diameter_x, diameter_y, diameter_z)

    def opening_get_soma_region_gpu(self, soma_region):
        soma_region_copy = soma_region.copy()
        radius = self.get_min_diameter_3d(soma_region)

        # on gpu
        # try:
        max_rate = 10
        soma_region_gpu = cp.array(soma_region)

        for i in range(max_rate):
            spherical_selem = ball(radius * (max_rate - i) / 10 / 2)
            spherical_selem_gpu = cp.array(spherical_selem)

            # 在 GPU 上执行 binary_opening
            # soma_region_res_gpu = binary_opening(soma_region_gpu, spherical_selem_gpu)
            soma_region_res_gpu = cupyx.scipy.ndimage.binary_opening(soma_region_gpu, spherical_selem_gpu)

            if soma_region_res_gpu.sum() == 0:
                continue

            soma_region_gpu = soma_region_res_gpu

        soma_region = cp.asnumpy(soma_region_gpu)
        del spherical_selem, radius, soma_region_res_gpu, soma_region_gpu
        # except:
        #     pass
        if (soma_region.sum() == 0):
            soma_region = soma_region_copy
        del soma_region_copy

        return soma_region

    def get_soma_region(self, img_path):
        # print(img_path, marker_path)
        in_tmp = img_path
        out_tmp = in_tmp.replace('.tif', '_gsdt.tif')

        if (sys.platform == "linux"):
            cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x gsdt -f gsdt -i {in_tmp} -o {out_tmp} -p 0 1 0 1.5'
            cmd_str = self.process_path(cmd_str)
            # print(cmd_str)
            subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
        else:
            cmd = f'{v3d_path} /x gsdt /f gsdt /i {in_tmp} /o {out_tmp} /p 0 1 0 1.5'
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        pred = tifffile.imread(img_path).astype("uint8")
        pred[pred <= 255 / 2] = 0
        pred[pred > 255 / 2] = 1

        gsdt = tifffile.imread(out_tmp).astype("uint8")
        gsdt = np.flip(gsdt, axis=1)
        if (os.path.exists(out_tmp)): os.remove(out_tmp)
        del out_tmp, in_tmp

        max_gsdt = np.max(gsdt)
        gsdt[gsdt <= max_gsdt / 2] = 0
        gsdt[gsdt > max_gsdt / 2] = 1

        gsdt = binary_dilation(gsdt, iterations=5).astype("uint8")
        soma_region = np.logical_and(pred, gsdt).astype("uint8")
        soma_region = self.dusting(soma_region)
        del pred, gsdt, max_gsdt

        soma_region, original_shape, min_coords = self.crop_nonzero(soma_region)
        # soma_region = opening_get_soma_region(soma_region)
        soma_region = self.opening_get_soma_region_gpu(soma_region)
        soma_region = self.dusting(soma_region)
        # restore original size
        soma_region = self.restore_original_size(soma_region, original_shape, min_coords)

        return soma_region

    def get_soma_regions_file(self, file_name, tif_folder, soma_folder):
        if ("gsdt" in file_name):
            os.remove(file_name)
            return
        tif_path = os.path.join(tif_folder, file_name)
        soma_region_path = os.path.join(soma_folder, os.path.splitext(file_name)[0] + '.tif')
        # muti_soma_marker_path = os.path.join(muti_soma_marker_folder, os.path.splitext(file_name)[0] + '.marker')
        # muti_soma_marker_path = find_muti_soma_marker_file(file_name, muti_soma_marker_folder)

        if (os.path.exists(soma_region_path)):
            return
        try:
            soma_region = self.get_soma_region(tif_path)
        except:
            return
        if (soma_region is None):
            return
        # binary
        soma_region = np.where(soma_region > 0, 1, 0).astype("uint8")
        tifffile.imwrite(soma_region_path, soma_region * 255, compression='zlib')

        del soma_region

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
import tifffile
from skimage.transform import resize
import os
import pandas as pd
from scipy.optimize import curve_fit

class SmartSomaRegionFinder():
    def find_resolution(self, df, filename):
        # print(filename)
        # filename = filename.split('.')[0]
        for i in range(len(df)):
            # print(df.iloc[i, 1], filename)
            if df.iloc[i, 1] == filename or df.iloc[i, 1] == filename.replace('.tif', ''):
                return df.iloc[i, 3]
        print(filename)
        exit()

    def exponential_decay(self, x, a, b, c):
        return a * np.exp(-b * (x - 2)) + c

    def calc_high_freq(self, seg, kernel_radii):
        # 存储高频能量占比和高频幅值平均值
        high_freq_ratios = []
        high_freq_averages = []

        for radius in kernel_radii:
            struct = morphology.ball(radius)

            # 形态学开运算
            opened_img = morphology.opening(seg, struct)
            opened_img = opened_img * seg

            if (np.sum(opened_img) == 0):
                break

            # 提取表面网格
            verts, faces, normals, values = measure.marching_cubes(opened_img, level=0)

            # 计算傅里叶描述子
            # 将三维表面网格展开为一维信号
            signal = verts.flatten()

            # 进行傅里叶变换
            F = np.fft.fft(signal)
            N = len(F)

            # 计算幅值谱
            F_magnitude = np.abs(F)

            # 总能量
            E_total = np.sum(F_magnitude ** 2)

            # 设定频率阈值（例如，前10%的频率作为低频）
            k_threshold = int(0.1 * N)

            # 高频能量
            E_high = np.sum(F_magnitude[k_threshold:] ** 2)

            # 高频能量占比
            R = E_high / E_total
            high_freq_ratios.append(R)

            # 平均高频幅值
            A_high = np.mean(F_magnitude[k_threshold:])
            high_freq_averages.append(A_high)

        return high_freq_ratios, high_freq_averages

    def calc_high_freq_gpu(self, seg, kernel_radii):
        # from cupyx.scipy.ndimage import binary_opening
        # 存储高频能量占比和高频幅值平均值
        high_freq_ratios = []
        high_freq_averages = []

        for radius in kernel_radii:
            # 使用 GPU 生成结构元素
            struct = morphology.ball(radius)

            # 将输入图像和结构元素转移到 GPU
            seg_gpu = cp.asarray(seg)
            struct_gpu = cp.asarray(struct)

            # GPU 上的形态学开运算
            opened_img_gpu = cupyx.scipy.ndimage.binary_opening(seg_gpu, struct_gpu)
            opened_img_gpu = opened_img_gpu * seg_gpu

            # 将结果转换回 CPU 以用于进一步处理
            opened_img = cp.asnumpy(opened_img_gpu)

            if (np.sum(opened_img) == 0):
                break

            # 提取表面网格
            verts, faces, normals, values = measure.marching_cubes(opened_img, level=0)

            # 计算傅里叶描述子
            # 将三维表面网格展开为一维信号
            signal = verts.flatten()

            # 使用 GPU 进行傅里叶变换
            signal_gpu = cp.asarray(signal)
            F_gpu = cp.fft.fft(signal_gpu)
            F_magnitude_gpu = cp.abs(F_gpu)

            # 转换为 CPU 数组
            F_magnitude = cp.asnumpy(F_magnitude_gpu)
            N = len(F_magnitude)

            # 总能量
            E_total = np.sum(F_magnitude ** 2)

            # 设定频率阈值（例如，前10%的频率作为低频）
            k_threshold = int(0.1 * N)

            # 高频能量
            E_high = np.sum(F_magnitude[k_threshold:] ** 2)

            # 高频能量占比
            R = E_high / E_total
            high_freq_ratios.append(R)

            # 平均高频幅值
            A_high = np.mean(F_magnitude[k_threshold:])
            high_freq_averages.append(A_high)

        return high_freq_ratios, high_freq_averages

    def largest_connected_component(self, binary_image):
        labeled_image, num_features = label(binary_image)
        component_sizes = np.bincount(labeled_image.ravel())
        component_sizes[0] = 0
        largest_component_label = np.argmax(component_sizes)
        largest_component = (labeled_image == largest_component_label).astype(np.uint8)

        return largest_component

    def test_soma_detectison(self, process_file_pair):
        seg_file, neuron_info_df, result_img_file = process_file_pair

        seg = tifffile.imread(seg_file).astype(np.uint8)

        resolution = self.find_resolution(neuron_info_df, os.path.basename(seg_file))
        resolution = resolution.split(', ')[1]
        resolution = (1, float(resolution), float(resolution))
        origin_img_size = seg.shape
        seg = resize(seg, (seg.shape[0] * resolution[0], seg.shape[1] * resolution[1], seg.shape[2] * resolution[2]),
                     order=0)
        seg = (seg - seg.min()) / (seg.max() - seg.min())
        seg = np.where(seg > 0, 1, 0).astype(np.uint8)
        # 填补空洞
        seg = ndimage.binary_fill_holes(seg).astype(int)

        # 定义核半径列表
        min_radii, max_radii = 2, 15
        kernel_radii = np.linspace(min_radii, max_radii, 20)

        high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)

        kernel_radii = kernel_radii[:len(high_freq_ratios)]
        if(len(kernel_radii) == 2):
            # kernel_radii = kernel_radii + [(kernel_radii[1] + kernel_radii[0]) / 2]
            kernel_radii = np.append(kernel_radii, (kernel_radii[1] + kernel_radii[0]) / 2)
            print(kernel_radii)
            high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)
            print(high_freq_averages, high_freq_ratios)
        elif(len(kernel_radii) == 1):
            # kernel_radii = kernel_radii + [kernel_radii[0] + 0.2, kernel_radii[0] - 0.2]
            kernel_radii = np.append(kernel_radii, kernel_radii[0] + 0.1)
            kernel_radii = np.append(kernel_radii, kernel_radii[0] + 0.2)
            print(kernel_radii)
            high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)
            print(high_freq_averages, high_freq_ratios)
        elif(len(kernel_radii) == 0):
            return None
        high_freq_averages = np.array(high_freq_averages)
        high_freq_averages = (high_freq_averages - high_freq_averages.min()) / (
                    high_freq_averages.max() - high_freq_averages.min())
        try:
            popt, pcov = curve_fit(self.exponential_decay, kernel_radii, high_freq_averages, p0=[1, 1, 0], maxfev=10000)
        except:
            return None

        # 生成拟合曲线数据
        x_fit = np.linspace(kernel_radii.min(), kernel_radii.max(), 100)
        y_fit = self.exponential_decay(x_fit, *popt)

        fitted_gradients = np.gradient(y_fit, x_fit)
        fitted_gradients = (fitted_gradients - fitted_gradients.min()) / (
                    fitted_gradients.max() - fitted_gradients.min()) - 1


        best_radius = x_fit[np.argmin(np.abs(fitted_gradients - 0.9))]

        result_open_img = morphology.opening(seg, morphology.ball(best_radius))
        result_open_img = result_open_img * seg
        result_img = resize(result_open_img, origin_img_size, order=0)
        result_img = self.largest_connected_component(result_img)
        result_img = np.where(result_img > 0, 1, 0).astype(np.uint8)
        return result_img

        # tifffile.imwrite(result_img_file, result_img, compression='zlib')



class SmartSomaRegionFinder_v2():
    def find_resolution(self, df, filename):
        # print(filename)
        # filename = filename.split('.')[0]
        for i in range(len(df)):
            # print(df.iloc[i, 1], filename)
            if df.iloc[i, 1] == filename or df.iloc[i, 1] == filename.replace('.tif', ''):
                return df.iloc[i, 3]
        print(filename)
        exit()

    def exponential_decay(self, x, a, b, c):
        return a * np.exp(-b * (x - 2)) + c

    def calc_high_freq(self, seg, kernel_radii):
        # 存储高频能量占比和高频幅值平均值
        high_freq_ratios = []
        high_freq_averages = []

        for radius in kernel_radii:
            struct = morphology.ball(radius)

            # 形态学开运算
            opened_img = morphology.opening(seg, struct)
            opened_img = opened_img * seg

            if (np.sum(opened_img) == 0):
                break

            # 提取表面网格
            verts, faces, normals, values = measure.marching_cubes(opened_img, level=0)

            # 计算傅里叶描述子
            # 将三维表面网格展开为一维信号
            signal = verts.flatten()

            # 进行傅里叶变换
            F = np.fft.fft(signal)
            N = len(F)

            # 计算幅值谱
            F_magnitude = np.abs(F)

            # 总能量
            E_total = np.sum(F_magnitude ** 2)

            # 设定频率阈值（例如，前10%的频率作为低频）
            k_threshold = int(0.1 * N)

            # 高频能量
            E_high = np.sum(F_magnitude[k_threshold:] ** 2)

            # 高频能量占比
            R = E_high / E_total
            high_freq_ratios.append(R)

            # 平均高频幅值
            A_high = np.mean(F_magnitude[k_threshold:])
            high_freq_averages.append(A_high)

        return high_freq_ratios, high_freq_averages

    def calc_high_freq_gpu(self, seg, kernel_radii):
        # from cupyx.scipy.ndimage import binary_opening
        # 存储高频能量占比和高频幅值平均值
        high_freq_ratios = []
        high_freq_averages = []

        for radius in kernel_radii:
            # 使用 GPU 生成结构元素
            struct = morphology.ball(radius)

            # 将输入图像和结构元素转移到 GPU
            seg_gpu = cp.asarray(seg)
            struct_gpu = cp.asarray(struct)

            # GPU 上的形态学开运算
            opened_img_gpu = cupyx.scipy.ndimage.binary_opening(seg_gpu, struct_gpu)
            opened_img_gpu = opened_img_gpu * seg_gpu

            # 将结果转换回 CPU 以用于进一步处理
            opened_img = cp.asnumpy(opened_img_gpu)

            if (np.sum(opened_img) == 0):
                break

            # 提取表面网格
            verts, faces, normals, values = measure.marching_cubes(opened_img, level=0)

            # 计算傅里叶描述子
            # 将三维表面网格展开为一维信号
            signal = verts.flatten()

            # 使用 GPU 进行傅里叶变换
            signal_gpu = cp.asarray(signal)
            F_gpu = cp.fft.fft(signal_gpu)
            F_magnitude_gpu = cp.abs(F_gpu)

            # 转换为 CPU 数组
            F_magnitude = cp.asnumpy(F_magnitude_gpu)
            N = len(F_magnitude)

            # 总能量
            E_total = np.sum(F_magnitude ** 2)

            # 设定频率阈值（例如，前10%的频率作为低频）
            k_threshold = int(0.1 * N)

            # 高频能量
            E_high = np.sum(F_magnitude[k_threshold:] ** 2)

            # 高频能量占比
            R = E_high / E_total
            high_freq_ratios.append(R)

            # 平均高频幅值
            A_high = np.mean(F_magnitude[k_threshold:])
            high_freq_averages.append(A_high)

        return high_freq_ratios, high_freq_averages

    def largest_connected_component(self, binary_image):
        labeled_image, num_features = label(binary_image)
        component_sizes = np.bincount(labeled_image.ravel())
        component_sizes[0] = 0
        largest_component_label = np.argmax(component_sizes)
        largest_component = (labeled_image == largest_component_label).astype(np.uint8)

        return largest_component

    def test_soma_detectison(self, seg, xy_resolution):
        resolution = (1, float(xy_resolution), float(xy_resolution))
        origin_img_size = seg.shape
        print("begin resize: ", seg.shape)
        seg = resize(seg, (seg.shape[0] * resolution[0], seg.shape[1] * resolution[1], seg.shape[2] * resolution[2]),
                     order=0)
        print("end resize: ", seg.shape)
        seg = (seg - seg.min()) / (seg.max() - seg.min())
        seg = np.where(seg > 0, 1, 0).astype(np.uint8)
        # 填补空洞
        seg = ndimage.binary_fill_holes(seg).astype(int)

        # 定义核半径列表
        min_radii, max_radii = 2, 15
        kernel_radii = np.linspace(min_radii, max_radii, 20)

        high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)

        kernel_radii = kernel_radii[:len(high_freq_ratios)]
        if(len(kernel_radii) == 2):
            # kernel_radii = kernel_radii + [(kernel_radii[1] + kernel_radii[0]) / 2]
            kernel_radii = np.append(kernel_radii, (kernel_radii[1] + kernel_radii[0]) / 2)
            print(kernel_radii)
            high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)
            print(high_freq_averages, high_freq_ratios)
        elif(len(kernel_radii) == 1):
            # kernel_radii = kernel_radii + [kernel_radii[0] + 0.2, kernel_radii[0] - 0.2]
            kernel_radii = np.append(kernel_radii, kernel_radii[0] + 0.1)
            kernel_radii = np.append(kernel_radii, kernel_radii[0] + 0.2)
            print(kernel_radii)
            high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)
            print(high_freq_averages, high_freq_ratios)
        elif(len(kernel_radii) == 0):
            return None
        high_freq_averages = np.array(high_freq_averages)
        high_freq_averages = (high_freq_averages - high_freq_averages.min()) / (
                    high_freq_averages.max() - high_freq_averages.min())
        try:
            popt, pcov = curve_fit(self.exponential_decay, kernel_radii, high_freq_averages, p0=[1, 1, 0], maxfev=10000)
        except:
            return None

        # 生成拟合曲线数据
        x_fit = np.linspace(kernel_radii.min(), kernel_radii.max(), 100)
        y_fit = self.exponential_decay(x_fit, *popt)

        fitted_gradients = np.gradient(y_fit, x_fit)
        fitted_gradients = (fitted_gradients - fitted_gradients.min()) / (
                    fitted_gradients.max() - fitted_gradients.min()) - 1


        best_radius = x_fit[np.argmin(np.abs(fitted_gradients - 0.9))]

        result_open_img = morphology.opening(seg, morphology.ball(best_radius))
        result_open_img = result_open_img * seg
        result_img = resize(result_open_img, origin_img_size, order=0)
        result_img = self.largest_connected_component(result_img)
        result_img = np.where(result_img > 0, 1, 0).astype(np.uint8)
        return result_img

        # tifffile.imwrite(result_img_file, result_img, compression='zlib')

