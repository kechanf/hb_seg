import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tifffile
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssim
import os
from skimage import exposure
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import cupy as cp
from cupyx.scipy.ndimage import zoom, gaussian_filter
import cupyx
from skimage.transform import resize
from torch.distributed.pipeline.sync.worker import worker

from nnUNet.scripts.mip import main_mip
from tqdm import tqdm


def get_max_connected_region(image):
    labels, num_features = ndi.label(image)
    sizes = ndi.sum(image, labels, range(num_features + 1))
    max_label = np.argmax(sizes)
    result = np.where(labels == max_label, 1, 0)
    return result.astype(np.float32)

def get_fluorescence_image(soma, sigma=15):
    soma_gaussian = ndi.gaussian_filter(soma, sigma=sigma)
    # soma_gaussian = (soma_gaussian - soma_gaussian.min()) / (soma_gaussian.max() - soma_gaussian.min())
    test_soma_gaussian = soma_gaussian.copy()
    test_soma_gaussian[soma == 0] = 0
    # print(np.max(test_soma_gaussian), np.min(test_soma_gaussian))

    soma_gaussian = soma_gaussian / soma_gaussian.max()

    soma_gaussian[soma > 0] = soma[soma > 0]
    # soma_gaussian = (soma_gaussian - soma_gaussian.min()) / (soma_gaussian.max() - soma_gaussian.min()).astype(
    #     np.float32)
    soma_gaussian = soma_gaussian.astype(np.float32)

    return soma_gaussian

def get_fluorescence_image_gpu(soma_gpu, sigma=25):
    soma_gaussian_gpu = gaussian_filter(soma_gpu, sigma=sigma)
    soma_gaussian_gpu /= soma_gaussian_gpu.max()
    soma_gaussian_gpu[cp.asarray(soma_gpu) > 0] = 1

    return soma_gaussian_gpu

def get_roi(soma):
    # 确保数据是二值的，通常我们处理的数据应该已经是二值化的
    binary_image = (soma > 0).astype(np.uint)

    # 查找非零元素的索引
    non_zero_indices = np.nonzero(binary_image)

    # 获取非零元素的形状
    if non_zero_indices[0].size > 0:
        # 计算非零体素块的边界框
        z_min, z_max = non_zero_indices[0].min(), non_zero_indices[0].max()
        y_min, y_max = non_zero_indices[1].min(), non_zero_indices[1].max()
        x_min, x_max = non_zero_indices[2].min(), non_zero_indices[2].max()
        # print(z_min, z_max, y_min, y_max, x_min, x_max)

        width = max(z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
        # width = (width*3, width*3, width*3)
        width = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
        width = (width[0]*2, width[1]*2, width[2]*2)
        # print(width)

        z_min, z_max = (z_min + z_max) / 2 - width[0] / 2, (z_min + z_max) / 2 + width[0] / 2
        y_min, y_max = (y_min + y_max) / 2 - width[1] / 2, (y_min + y_max) / 2 + width[1] / 2
        x_min, x_max = (x_min + x_max) / 2 - width[2] / 2, (x_min + x_max) / 2 + width[2] / 2
        # print(z_min, z_max, y_min, y_max, x_min, x_max)

        z_min, z_max = max(0, int(z_min)), min(binary_image.shape[0], int(z_max))
        y_min, y_max = max(0, int(y_min)), min(binary_image.shape[1], int(y_max))
        x_min, x_max = max(0, int(x_min)), min(binary_image.shape[2], int(x_max))
        # print(binary_image.shape)

        # non_zero_shape = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
        # print("Non-zero voxel block shape:", non_zero_shape)
        # print("ROI shape:", (z_max - z_min, y_max - y_min, x_max - x_min))
        return (z_min, z_max, y_min, y_max, x_min, x_max)
    else:
        print("No non-zero elements found in the image.")

def get_roi_gpu(soma):
    binary_image = (soma > 0).astype(cp.uint)
    non_zero_indices = cp.nonzero(binary_image)
    if non_zero_indices[0].size > 0:
        z_min, z_max = non_zero_indices[0].min(), non_zero_indices[0].max()
        y_min, y_max = non_zero_indices[1].min(), non_zero_indices[1].max()
        x_min, x_max = non_zero_indices[2].min(), non_zero_indices[2].max()

        width = max(z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
        width = (width*3, width*3, width*3)

        z_min, z_max = (z_min + z_max) / 2 - width[0] / 2, (z_min + z_max) / 2 + width[0] / 2
        y_min, y_max = (y_min + y_max) / 2 - width[1] / 2, (y_min + y_max) / 2 + width[1] / 2
        x_min, x_max = (x_min + x_max) / 2 - width[2] / 2, (x_min + x_max) / 2 + width[2] / 2

        z_min, z_max = max(0, int(z_min)), min(binary_image.shape[0], int(z_max))
        y_min, y_max = max(0, int(y_min)), min(binary_image.shape[1], int(y_max))
        x_min, x_max = max(0, int(x_min)), min(binary_image.shape[2], int(x_max))

        return (z_min, z_max, y_min, y_max, x_min, x_max)
    else:
        print("No non-zero elements found in the image.")




def find_best_fluorescence_image(img, soma, tol=0.1, max_iter=100, low=10, high=30):
    origin_img = img
    roi_pos = get_roi(get_max_connected_region(np.where(soma > 0.9, 1, 0)))
    roi_img = img[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3], roi_pos[4]:roi_pos[5]]
    roi_soma = soma[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3], roi_pos[4]:roi_pos[5]]
    # print(np.max(img), np.min(img))
    # print(np.max(soma), np.min(soma))

    """    max_ssim = 0
    soma_gaussian_best = None
    for sigma in range(5, 50):
        soma_gaussian = get_fluorescence_image(soma, sigma)
        ssim_index = ssim(soma_gaussian, img, data_range=soma.max() - soma.min())
        print(sigma, ssim_index)
        if ssim_index > max_ssim:
            max_ssim = ssim_index
            soma_gaussian_best = soma_gaussian
    return img, soma_gaussian_best"""
    # return img, soma

    max_ssim = 0
    soma_gaussian_best = None
    best_sigma = None

    data_range = 1.0

    for sigma in range(5, 25):
        soma_gaussian = get_fluorescence_image(roi_soma, sigma)
        ssim_index = ssim(soma_gaussian, roi_img, data_range=data_range)
        # print(sigma, ssim_index)
        if ssim_index > max_ssim:
            max_ssim = ssim_index
            soma_gaussian_best = soma_gaussian
            best_sigma = sigma

    print(f"Best Sigma: {best_sigma}, Max SSIM: {max_ssim}")
    de_flu_img = roi_img + roi_soma - soma_gaussian_best * 0.3
    de_flu_img = np.clip(de_flu_img, 0, 1)

    result_img = origin_img.copy()
    result_img[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3], roi_pos[4]:roi_pos[5]] = de_flu_img
    result_img = np.clip(result_img, 0, 1)

    # equalizeHist
    result_img = exposure.equalize_adapthist(result_img, clip_limit=0.02, nbins=256)

    return roi_img, de_flu_img, result_img, best_sigma

def find_best_fluorescence_image_gpu(img, soma, tol=1, max_iter=100, low=5, high=50):
    origin_img = img
    roi_pos = get_roi_gpu(get_max_connected_region(soma))
    roi_img = img[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3], roi_pos[4]:roi_pos[5]]
    roi_soma = soma[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3], roi_pos[4]:roi_pos[5]]

    max_ssim = 0
    soma_gaussian_best = None
    best_sigma = None

    data_range = 1.0

    for sigma in range(5, 6):
        soma_gaussian = get_fluorescence_image_gpu(roi_soma, sigma)
        ssim_index = ssim(soma_gaussian, roi_img, data_range=data_range)
        # print(sigma, ssim_index)
        if ssim_index > max_ssim:
            max_ssim = ssim_index
            soma_gaussian_best = soma_gaussian
            best_sigma = sigma

    print(f"Best Sigma: {best_sigma}, Max SSIM: {max_ssim}")
    de_flu_img = roi_img + roi_soma - soma_gaussian_best * 1.0
    de_flu_img = np.clip(de_flu_img, 0, 1)

    result_img = origin_img.copy()
    result_img[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3], roi_pos[4]:roi_pos[5]] = de_flu_img
    result_img = np.clip(result_img, 0, 1)

    # equalizeHist
    result_img = exposure.equalize_adapthist(result_img, clip_limit=0.02, nbins=256)

    return roi_img, de_flu_img, result_img, best_sigma

def find_best_sigma_map(img, soma, tol=0.1, max_iter=100, low=5, high=50):
    # 如果soma为0
    if np.max(soma) == 0:
        best_sigma_map = np.zeros_like(soma)
        return best_sigma_map, 0

    roi_pos = get_roi(get_max_connected_region(np.where(soma > 0.9, 1, 0)))
    roi_img = img[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3], roi_pos[4]:roi_pos[5]]
    roi_soma = soma[roi_pos[0]:roi_pos[1], roi_pos[2]:roi_pos[3], roi_pos[4]:roi_pos[5]]
    # print(np.max(img), np.min(img))
    # print(np.max(soma), np.min(soma))

    # print(roi_img.shape, roi_soma.shape)

    """    max_ssim = 0
    soma_gaussian_best = None
    for sigma in range(5, 50):
        soma_gaussian = get_fluorescence_image(soma, sigma)
        ssim_index = ssim(soma_gaussian, img, data_range=soma.max() - soma.min())
        print(sigma, ssim_index)
        if ssim_index > max_ssim:
            max_ssim = ssim_index
            soma_gaussian_best = soma_gaussian
    return img, soma_gaussian_best"""
    # return img, soma

    max_ssim = 0
    soma_gaussian_best = None
    best_sigma = None

    data_range = 1.0

    for sigma in range(2, 25):
        soma_gaussian = get_fluorescence_image(roi_soma, sigma)
        try:
            ssim_index = ssim(soma_gaussian, roi_img, data_range=data_range)
        except:
            best_sigma=0
            break
        # ssim_index = ssim(soma_gaussian, roi_img, data_range=data_range)
        # print(sigma, ssim_index)
        if ssim_index > max_ssim:
            max_ssim = ssim_index
            soma_gaussian_best = soma_gaussian
            best_sigma = sigma

    print(f"Best Sigma: {best_sigma}, Max SSIM: {max_ssim}")

    # sigma_map = ndi.gaussian_filter(soma, sigma=best_sigma)
    sigma_map = get_fluorescence_image(soma, best_sigma)

    return sigma_map, best_sigma

def deflu_gamma(data, power_values, data_range = [0.5, 1.0]):
    power_values = ((power_values - power_values.min()) /
                    (power_values.max() - power_values.min()) *
                    (data_range[1] - data_range[0]) + data_range[0])


    # Normalize data
    min_data = np.min(data)
    max_data = np.max(data)
    data_normalized = (data - min_data) / (max_data - min_data)


    # Apply gamma correction
    data_corrected = data_normalized ** power_values

    data = data_corrected

    # Scale data back to original range
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255

    return data.astype("uint8")

def rescaled_gamma(data, power_values, data_range = [0.5, 1.0]):
    power_values = (
            (power_values - power_values.min()) /
            (power_values.max() - power_values.min()) *
                    (data_range[1] - data_range[0])
            + data_range[0]
    )


    # Normalize data
    min_data = np.min(data)
    max_data = np.max(data)
    data_normalized = (data - min_data) / (max_data - min_data)


    # Apply gamma correction
    data_corrected = data_normalized ** power_values
    data = data_corrected
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    return data


def gamma_t(data, power_value=0.5):
    power_values = np.ones(data.shape) * power_value
    min_data = np.min(data)
    max_data = np.max(data)
    data_normalized = (data - min_data) / (max_data - min_data)

    # Apply gamma correction
    data_corrected = data_normalized ** power_values

    data = data_corrected

    # Scale data back to original range
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255

    return data.astype("uint8")


def diffusion_adaptive_gamma(img, xy_resolution):
    origin_img = img.copy()
    origin_img_shape = img.shape

    img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
    resolution = (1, xy_resolution / 1000, xy_resolution / 1000)
    img = resize(img, (img.shape[0] * resolution[0], img.shape[1] * resolution[1], img.shape[2] * resolution[2]),
                 order=3)
    # img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)

    soma = np.where(img > 0.9, img, 0).astype(np.float32)
    # print(np.max(soma), np.min(soma), soma.shape)

    best_sigma_map, best_sigma = find_best_sigma_map(img, soma)
    best_sigma_map = resize(best_sigma_map, origin_img_shape, order=3, preserve_range=True, anti_aliasing=False).astype(
        best_sigma_map.dtype)
    result_img = rescaled_gamma(origin_img, best_sigma_map)
    # soma = resize(soma, origin_img_shape, order=3, preserve_range=True, anti_aliasing=False).astype(soma.dtype)

    # result_img = ((result_img - result_img.min()) / (result_img.max() - result_img.min()) * 255).astype("uint8")

    return result_img, best_sigma_map

def truncated_gamma(img, gamma_map, power_range=(0.5, 1.0)):
    img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
    # rescale gamma map
    if(power_range[0] == power_range[1]):
        gamma_map = power_range[0] * np.ones_like(gamma_map)
        print("fuck")
    elif(gamma_map.max() == gamma_map.min()):
        gamma_map = power_range[0] * np.ones_like(gamma_map)
        print("fuck")
    else:
        gamma_map = ((gamma_map - gamma_map.min()) / (gamma_map.max() - gamma_map.min()) * (power_range[1] - power_range[0]) + power_range[0])
    print(np.max(gamma_map), np.min(gamma_map))
    theta_map = np.exp(np.log(gamma_map) / (1 - gamma_map + 1e-6))
    mask = img < theta_map
    img[mask] = np.power(img[mask], gamma_map[mask])
    img[~mask] = img[~mask] - theta_map[~mask] + np.power(theta_map[~mask], gamma_map[~mask])

    img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
    return img

def truncated_diff_gamma(img, xy_resolution):
    origin_img = img.copy()
    origin_img_shape = img.shape

    img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
    resolution = (1, xy_resolution / 1000, xy_resolution / 1000)
    img = resize(img, (img.shape[0] * resolution[0], img.shape[1] * resolution[1], img.shape[2] * resolution[2]),
                 order=3)
    # img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)

    soma = np.where(img > 0.9, img, 0).astype(np.float32)
    # print(np.max(soma), np.min(soma), soma.shape)

    best_sigma_map, best_sigma = find_best_sigma_map(img, soma)
    best_sigma_map = resize(best_sigma_map, origin_img_shape, order=3, preserve_range=True, anti_aliasing=False).astype(
        best_sigma_map.dtype)
    result_img = rescaled_gamma(origin_img, best_sigma_map)
    # soma = resize(soma, origin_img_shape, order=3, preserve_range=True, anti_aliasing=False).astype(soma.dtype)

    # result_img = ((result_img - result_img.min()) / (result_img.max() - result_img.min()) * 255).astype("uint8")

    result_img = truncated_gamma(result_img, best_sigma_map)
    result_img = (result_img - result_img.min()) / (result_img.max() - result_img.min())
    return result_img



def visualize_result(img_file, xy_resolution, result_img_file, mip_file):
    # img_file = r"D:\tracing_ws\example_mips\raw\2385.tif"
    if (os.path.exists(result_img_file)):
        return img_file, 0
    else:
        pass

    img = tifffile.imread(img_file)
    origin_img = img.copy()
    origin_img_shape = img.shape

    if(xy_resolution == None):
        return img_file, None
    img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)
    resolution = (1, xy_resolution/1000, xy_resolution/1000)
    img = resize(img, (img.shape[0] * resolution[0], img.shape[1] * resolution[1], img.shape[2] * resolution[2]),
                 order=3)
    # img = (img - img.min()) / (img.max() - img.min()).astype(np.float32)

    soma = np.where(img > 0.9, img, 0).astype(np.float32)

    best_sigma_map, best_sigma = find_best_sigma_map(img, soma)
    best_sigma_map = resize(best_sigma_map, origin_img_shape, order=3, preserve_range=True, anti_aliasing=False).astype(best_sigma_map.dtype)
    # tifffile.imwrite(result_img_file.replace('.tif', '_gamma_map.tif'), best_sigma_map)
    result_img = deflu_gamma(origin_img, best_sigma_map)

    result_img = ((result_img - result_img.min()) / (result_img.max() - result_img.min()) * 255).astype("uint8")

    if(os.path.exists(result_img_file)):
        os.remove(result_img_file)
    tifffile.imwrite(result_img_file, result_img)

    #
    # img_gaussian_mip = np.max(img_gaussian, axis=0)
    # soma_mip = np.max(soma, axis=0)
    # best_sigma_map_mip = np.max(best_sigma_map, axis=0)
    # result_img_mip = np.max(result_img, axis=0)
    # equ_img_mip = np.max(equ_img, axis=0)
    # simple_gamma_img_mip = np.max(simple_gamma_img, axis=0)
    #
    # fig, ax = plt.subplots(1, 6, figsize=(12, 6))
    # ax[0].imshow(img_gaussian_mip, cmap='gray')
    # ax[0].set_title('Original Image')
    # ax[0].axis('off')
    # ax[1].imshow(soma_mip, cmap='gray')
    # ax[1].set_title('Soma region thed=.9')
    # ax[1].axis('off')
    # ax[2].imshow(best_sigma_map_mip, cmap='gray')
    # ax[2].set_title('Best Sigma Map')
    # ax[2].axis('off')
    # ax[3].imshow(result_img_mip, cmap='gray')
    # ax[3].set_title('Result Image')
    # ax[3].axis('off')
    # ax[4].imshow(equ_img_mip, cmap='gray')
    # ax[4].set_title('Equalized Image')
    # ax[4].axis('off')
    # ax[5].imshow(simple_gamma_img_mip, cmap='gray')
    # ax[5].set_title('Simple Gamma Image')
    # ax[5].axis('off')
    # plt.tight_layout()
    # # plt.show()
    # if(os.path.exists(mip_file)):
    #     os.remove(mip_file)
    # plt.savefig(mip_file)
    # plt.close()

    return img_file, best_sigma

#
# def visualize_result_gpu(img_file, xy_resolution, result_img_file, mip_file):
#     # img_file = r"D:\tracing_ws\example_mips\raw\2385.tif"
#
#     img = tifffile.imread(img_file)
#     img_gaussian = img.copy()
#     # to gpu
#     img = cp.asarray(img)
#     img = (img - img.min()) / (img.max() - img.min()).astype(cp.float32)
#     resolution = (1, xy_resolution/1000, xy_resolution/1000)
#     img = zoom(img, resolution, order=3)
#     soma = cp.where(img > 0.9, 1, 0).astype(cp.float32)
#
#     ion_img, de_flu_img, result_img, best_sigma = find_best_fluorescence_image_gpu(img, soma)
#     result_img = zoom(result_img, (1/resolution[0], 1/resolution[1], 1/resolution[2]), order=3)
#
#     result_img = cp.asnumpy(result_img)
#     result_img = ((result_img - result_img.min()) / (result_img.max() - result_img.min()) * 255).astype("uint8")
#     tifffile.imwrite(result_img_file, result_img)
#
#
#     img_gaussian_mip = np.max(img_gaussian, axis=0)
#     soma_mip = np.max(soma, axis=0)
#     ion_img_mip = np.max(cp.asnumpy(ion_img), axis=0)
#     ion_soma_mip = np.max(cp.asnumpy(de_flu_img), axis=0)
#     result_img_mip = np.max(result_img, axis=0)
#
#     fig, ax = plt.subplots(1, 5, figsize=(12, 6))
#     ax[0].imshow(img_gaussian_mip, cmap='gray')
#     ax[0].set_title('Original Image')
#     ax[0].axis('off')
#     ax[1].imshow(soma_mip, cmap='gray')
#     ax[1].set_title('Soma region thed=.9')
#     ax[1].axis('off')
#     ax[2].imshow(ion_img_mip, cmap='gray')
#     ax[2].set_title('ROI Image')
#     ax[2].axis('off')
#     ax[3].imshow(ion_soma_mip, cmap='gray')
#     ax[3].set_title('De-fluorescence ROI Image')
#     ax[3].axis('off')
#     ax[4].imshow(result_img_mip, cmap='gray')
#     ax[4].set_title('Result Image')
#     ax[4].axis('off')
#     plt.tight_layout()
#     # plt.show()
#     if(os.path.exists(mip_file)):
#         os.remove(mip_file)
#     plt.savefig(mip_file)
#     plt.close()
#
#     return best_sigma

def find_resolution(df, filename):
    # print(filename)
    filename = int(filename.split('_')[0].split('.')[0])
    return df[df['Cell ID'] == filename]['xy拍摄分辨率(*10e-3μm/px)'].values[0]
    # for i in range(len(df)):
    #     if int(df.iloc[i, 0]) == filename:
    #         return df.iloc[i, 43]
    # return None

def main_for_1342():
    img_dir = r"/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/raw"
    neuron_info_file = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
    result_img_dir = r"/data/kfchen/trace_ws/de_flu_test/de_tif_v2"
    mip_dir = r"/data/kfchen/trace_ws/de_flu_test/mip_v2"

    if (not os.path.exists(result_img_dir)):
        os.makedirs(result_img_dir)
    if (not os.path.exists(mip_dir)):
        os.makedirs(mip_dir)

    imgs = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    result_imgs = [f for f in os.listdir(result_img_dir) if f.endswith('.tif')]
    imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))
    result_imgs = sorted(result_imgs, key=lambda x: int(x.split('.')[0]))

    df = pd.read_csv(neuron_info_file, encoding='gbk')
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]

    # img_files = [f for f in img_files if "2455" in f]

    img_files = sorted(img_files, key=lambda x: int(x.split('.')[0]))
    # img_files = img_files[200:]
    # img_files = img_files[:50]
    # interested = ['3256']
    # img_files = [f for f in img_files if f.split('.')[0] in interested]
    # print(img_files)

    xy_resolution_list, mip_files, result_img_files = [], [], []
    for img_file in img_files:
        xy_resolution = find_resolution(df, img_file)
        mip_file = os.path.join(mip_dir, img_file.replace('.tif', '.png'))
        result_img_file = os.path.join(result_img_dir, img_file)
        xy_resolution_list.append(xy_resolution)
        mip_files.append(mip_file)
        result_img_files.append(result_img_file)
    img_files = [os.path.join(img_dir, f) for f in img_files]

    # sigma_list = []
    result_list = []

    # 使用线程池执行多线程任务
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 创建一个future列表，用于保持结果的顺序
        futures = [executor.submit(visualize_result, img_file, xy_resolution, result_img_file, mip_file) for img_file, xy_resolution, result_img_file, mip_file in zip(img_files, xy_resolution_list, result_img_files, mip_files)]

        # 按照任务被提交的顺序获取结果
        for future in as_completed(futures):
            img_file, sigma = future.result()
            result_list.append((img_file, sigma))

    # for img_file, xy_resolution, result_img_file, mip_file in zip(img_files, xy_resolution_list, result_img_files,
    #                                                               mip_files):
    #     if (xy_resolution == None):
    #         result_list.append((img_file, 0))
    #     _, sigma = visualize_result(os.path.join(img_dir, img_file), xy_resolution, result_img_file, mip_file)
    #     # sigma_list.append(sigma)
    #     result_list.append((img_file, sigma))

    # to csv img file and sigma
    df = pd.DataFrame({'img': [r.split('/')[-1] for r in img_files], 'sigma': [r[1] for r in result_list]})
    df.to_csv('/data/kfchen/trace_ws/de_flu_test/sigma.csv', index=False)

from pylib.file_io import load_image
def prepare_data_14k():
    v3d_img_root = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"
    v3d_img_dirs = [f for f in os.listdir(v3d_img_root) if f.startswith('human_brain_data_v3draw')]
    v3d_imgs = []

    final_recon_info_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv"
    final_recon_info = pd.read_csv(final_recon_info_file)
    ids = final_recon_info['id'].values
    ids.append("14002")

    for v3d_img_dir in v3d_img_dirs:
        # walk
        v3d_img_dir = os.path.join(v3d_img_root, v3d_img_dir)
        for root, dirs, files in os.walk(v3d_img_dir):
            for file in files:
                if file.endswith('.v3draw'):
                    current_id = int(file.split('_')[0])
                    if(current_id in ids):
                        v3d_imgs.append(os.path.join(root, file))

    # v3d_imgs = v3d_imgs[:10]
    tif_dir = "/data/kfchen/trace_ws/de_flu_test/14k_tif"
    # for v3d_img in v3d_imgs:
    #     tif_file = os.path.join(tif_dir, os.path.basename(v3d_img).replace('.v3draw', '.tif'))
    #     img = load_image(v3d_img)
    #     img = np.array(img).astype(np.float32)
    #     img = (img - img.min()) / (img.max() - img.min())
    #     img = (img * 255).astype(np.uint8)
    #     tifffile.imwrite(tif_file, img)

    def current_task(v3d_img, tif_dir):
        tif_file = os.path.join(tif_dir, os.path.basename(v3d_img).replace('.v3draw', '.tif'))
        img = load_image(v3d_img)[0]
        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        tifffile.imwrite(tif_file, img)

    with tqdm(total=len(v3d_imgs), desc="Processing images") as pbar:
        # 使用 ThreadPoolExecutor 来管理多线程
        with ThreadPoolExecutor() as executor:
            futures = []
            for v3d_img in v3d_imgs:
                futures.append(executor.submit(current_task, v3d_img, tif_dir))

            # 更新进度条，当一个任务完成时
            for future in futures:
                future.result()  # 阻塞直到当前线程完成
                pbar.update(1)  # 更新进度条


def main_for_14k():
    img_dir = r'/data/kfchen/trace_ws/de_flu_test/14k_tif'
    if(not os.path.exists(img_dir)):
        os.makedirs(img_dir)
        prepare_data_14k()
    # exit()
    neuron_info_file = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"

    result_img_dir = r"/data/kfchen/trace_ws/de_flu_test/de_tif_v2_14k"
    mip_dir = r"/data/kfchen/trace_ws/de_flu_test/mip_v2_14k"

    if (not os.path.exists(result_img_dir)):
        os.makedirs(result_img_dir)
    if (not os.path.exists(mip_dir)):
        os.makedirs(mip_dir)

    df = pd.read_csv(neuron_info_file, encoding='gbk')
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    # img_files = [f if '3575' in f else None for f in img_files]
    # img_files = [f for f in img_files if f != None]
    img_files = sorted(img_files, key=lambda x: int(x.split('_')[0]))
    # img_files = img_files[:50]
    # img_files = [f for f in img_files if int(f.split('_')[0]) == 3294]

    xy_resolution_list, mip_files, result_img_files = [], [], []
    for img_file in img_files:
        xy_resolution = find_resolution(df, img_file)
        mip_file = os.path.join(mip_dir, img_file.replace('.tif', '.png'))
        result_img_file = os.path.join(result_img_dir, img_file)
        xy_resolution_list.append(xy_resolution)
        mip_files.append(mip_file)
        result_img_files.append(result_img_file)

    # result_list = []

    # pbar = tqdm(total=len(img_files), desc="Processing images")
    # for img_file, xy_resolution, result_img_file, mip_file in zip(img_files, xy_resolution_list, result_img_files,
    #                                                               mip_files):
    #     if(not os.path.exists(result_img_file)):
    #         visualize_result(os.path.join(img_dir, img_file), xy_resolution, result_img_file, mip_file)
    #     pbar.update(1)

    def current_task(img_file, xy_resolution, result_img_file, mip_file):
        visualize_result(os.path.join(img_dir, img_file), xy_resolution, result_img_file, mip_file)


    with tqdm(total=len(img_files), desc="Processing images") as pbar:
        # 使用 ThreadPoolExecutor 来管理多线程
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for img_file, xy_resolution, result_img_file, mip_file in zip(img_files, xy_resolution_list, result_img_files, mip_files):
                futures.append(executor.submit(current_task, img_file, xy_resolution, result_img_file, mip_file))

            # 更新进度条，当一个任务完成时
            for future in futures:
                future.result()
                pbar.update(1)  # 更新进度条


    # to csv img file and sigma
    # df = pd.DataFrame({'img': [r.split('/')[-1] for r in img_files], 'sigma': [r[1] for r in result_list]})
    # df.to_csv('/data/kfchen/trace_ws/de_flu_test_14k/sigma.csv', index=False)

if __name__ == '__main__':
    # main_for_1342()
    main_for_14k()

