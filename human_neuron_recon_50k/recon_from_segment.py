import os
import shutil
import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
import numpy as np
import skimage.morphology
import tifffile
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk
import sys


from simple_swc_tool.swc_io import read_swc, write_swc
from scipy import ndimage
import pandas as pd
import os
import networkx as nx
import inspect
import tempfile
from scipy.spatial import ConvexHull
from skimage.draw import polygon, ellipsoid
from simple_swc_tool.get_soma_region_from_seg import SmartSomaRegionFinder_v2
import cProfile
import pstats

v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"
MAX_PROCESSES = 16

class FileProcessingStep:
    def __init__(self, step_name, process_function, file_name, input_dirs, output_dirs=None, supplementary_files=None):
        self.step_name = step_name
        self.process_function = process_function  # Function that performs the processing
        self.file_name = file_name  # Common file name for inputs and outputs
        self.input_dirs = input_dirs  # List of directories containing input files
        self.output_dirs = output_dirs  # Directory to store the output files
        self.supplementary_files = supplementary_files if supplementary_files else []  # List of supplementary files

    def count_function_parameters(self, function):
        signature = inspect.signature(function)
        return len(signature.parameters)

    def execute(self):
        # Create output directory if it doesn't exist
        for output_dir in self.output_dirs:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        # Collect input files with the specified file_name from input directories
        input_files = [os.path.join(input_dir, self.file_name) for input_dir in self.input_dirs]
        for i in range(len(input_files)):
            if(self.input_dirs[i].endswith('swc')):
                input_files[i] = input_files[i].replace('.tif', '.swc')
            elif(self.input_dirs[i].endswith('marker')):
                input_files[i] = input_files[i].replace('.tif', '.marker')

        output_files = [os.path.join(output_dir, self.file_name) for output_dir in self.output_dirs]
        for i in range(len(output_files)):
            if(self.output_dirs[i].endswith('swc')):
                output_files[i] = output_files[i].replace('.tif', '.swc')
            elif(self.output_dirs[i].endswith('marker')):
                output_files[i] = output_files[i].replace('.tif', '.marker')

        if(self.count_function_parameters(self.process_function) == 2):
            self.process_function(input_files, output_files)
        elif(self.count_function_parameters(self.process_function) == 3):
            self.process_function(input_files, output_files, self.supplementary_files)
        else:
            raise ValueError("The number of parameters of the process function must be 2 or 3.")


class FileProcessingPipeline:
    def __init__(self, root_dir, file_name, meta_info=None):
        if not root_dir:
            raise ValueError("Root directory must be specified.")
        self.root_dir = root_dir
        self.steps = []
        self.file_name = file_name
        self.meta_info = meta_info

    def add_step(self, step_name, process_function, input_steps, output_steps, meta_info=None):
        # Define the output directory for the step
        # output_dirs = os.path.join(self.root_dir, step_name)
        input_dirs = [os.path.join(self.root_dir, input_step) for input_step in input_steps]
        output_dirs = [os.path.join(self.root_dir, output_step) for output_step in output_steps]
        step = FileProcessingStep(step_name, process_function, self.file_name, input_dirs, output_dirs,
                                  meta_info)
        self.steps.append(step)

    def run(self):
        if not self.steps:
            print("No steps to run in the pipeline.")
            return

        for step in self.steps:
            step.execute()


def copy_file_from_seg_result(origin_seg_file, seg_dir, name_mapping_file):
    def get_full_name(file_name, df):
        full_name = df[df['nnunet_name'] == file_name]['full_name']
        result = str(full_name.values[0])
        if(not result.endswith('.tif')):
            result += '.tif'
        return result
    # print(origin_seg_file)
    if((not origin_seg_file.endswith('.tif')) and (not origin_seg_file.endswith('.nii.gz'))):
        return

    if(not os.path.exists(seg_dir)):
        os.makedirs(seg_dir, exist_ok=True)

    # copy the seg file to the output directory
    file_name = os.path.basename(origin_seg_file)
    if (file_name.endswith('.nii.gz')):
        seg = sitk.ReadImage(origin_seg_file)
        seg = sitk.GetArrayFromImage(seg)
        seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
        file_name = file_name.replace('_pred0.nii.gz', '.tif')
        tifffile.imwrite(os.path.join(seg_dir, file_name), seg, compression='zlib')
    else:
        seg = tifffile.imread(origin_seg_file)
        seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
        tifffile.imwrite(os.path.join(seg_dir, file_name), seg, compression='zlib')

    df = pd.read_csv(name_mapping_file)
    full_name = get_full_name(file_name.split('.')[0], df)
    if full_name is None:
        print(f"Error: {file_name} not found in the name mapping file.")
        return
    # rename the seg file
    new_file_path = os.path.join(seg_dir, full_name)
    os.rename(os.path.join(seg_dir, file_name), new_file_path)
    # print(new_file_path, "ok")


def prepare_seg_files(origin_seg_dir, seg_dir, name_mapping_file):
    origin_seg_files = [os.path.join(origin_seg_dir, f) for f in os.listdir(origin_seg_dir)]
    # for origin_seg_file in origin_seg_files:
    #     copy_file_from_seg_result(origin_seg_file, seg_dir, name_mapping_file)

    # 多线程
    pool = Pool(MAX_PROCESSES)
    func = partial(copy_file_from_seg_result, seg_dir=seg_dir, name_mapping_file=name_mapping_file)
    pool.map(func, origin_seg_files)
    pool.close()
    pool.join()

class AutoTracePipeline(FileProcessingPipeline):
    def __init__(self, root_dir, file_name, meta_info=None):
        if(meta_info is None):
            # warning
            print("Warning: meta_info is empty. Default values will be used.")

            meta_info = {
                "xy_resolution": 500,
                "z_resolution": 1000,
                "soma_x": 0,
                "soma_y": 0,
                "soma_z": 0,
            }
            # to df
            meta_info = pd.DataFrame([meta_info])
        super().__init__(root_dir, file_name, meta_info)

        self.add_step("1_skel_seg", self.skel_seg, ["0_seg"], ["1_skel_seg"])
        self.add_step('2_soma_region', self.get_soma_region, ['0_seg'], ['2_soma_region', "4_soma_marker"])
        self.add_step("3_skel_with_soma", self.skel_with_soma, ["1_skel_seg", "2_soma_region"], ["3_skel_with_soma"])

        self.add_step("5_swc", self.trace_app2_with_soma, ["3_skel_with_soma", "4_soma_marker"], ["5_swc"])
        # self.add_step("5_swc", self.trace_gtree, ["3_skel_with_soma"], ["5_swc"])

        # self.add_step("6_reconnect_soma", self.connect_to_soma_file, ["5_swc", "2_soma_region"], ["6_connect_soma_swc"])

        # self.add_step("7_rescale_xy_resolution", self.rescale_xy_resolution, ["6_connect_soma_swc"], ["7_scaled_1um_swc"])
        self.add_step("7_rescale_xy_resolution", self.rescale_xy_resolution, ["5_swc"], ["7_scaled_1um_swc"])

        # self.add_step("8_estimated_radius_swc", self.get_estimated_radius, ["7_scaled_1um_swc", "0_seg"], ["8_estimated_radius_swc"])

    def find_resolution(self):
        return self.meta_info['xy_resolution']

    def skel_seg(self, input_files, output_files):
        if(not input_files):
            return
        seg_file = input_files[0]
        skel_file = output_files[0]

        if ((not os.path.exists(seg_file)) or os.path.exists(skel_file)):
            return

        data = tifffile.imread(seg_file).astype("uint8")
        # data = np.flip(data, axis=1)

        skel = skeletonize_3d(data).astype("uint8")
        tifffile.imwrite(skel_file, skel * 255, compression='zlib')

    def get_soma_region(self, input_files, output_files):
        def clear_gsdt_file(seg_file):
            input_dir = os.path.dirname(seg_file)
            for f in os.listdir(input_dir):
                if 'gsdt' in f and os.path.basename(seg_file).split('.')[0] in f:
                    os.remove(os.path.join(input_dir, f))

        def compute_centroid(mask):
            # 计算三维 mask 的重心
            labeled_mask = skimage.measure.label(mask)
            props = regionprops(labeled_mask)

            if len(props) > 0:
                # 获取第一个区域的重心坐标
                centroid = props[0].centroid
                return centroid
            else:
                return None

        def find_solid_convex_hull_3d(binary_image, seed_point):
            # 确保种子点在前景内
            if not binary_image[seed_point]:
                print(f"Seed point: {seed_point}")
                raise ValueError("Seed point is not in the foreground.")

            # 获取图像中所有前景点的坐标
            foreground_coords = np.argwhere(binary_image)

            # 在前景点中找到种子点
            if seed_point not in foreground_coords:
                print(f"Seed point: {seed_point}")
                raise ValueError("Seed point not in the foreground.")

            # 使用ConvexHull来计算包含种子点的最大凸包
            hull = ConvexHull(foreground_coords)

            # 获取凸包的所有顶点（3D坐标）
            convex_hull_points = foreground_coords[hull.vertices]

            # 创建一个与输入图像相同形状的空白图像
            solid_convex_hull = np.zeros_like(binary_image)

            # 使用多边形面填充凸包的内部
            for simplex in hull.simplices:
                rr, cc, zz = polygon(convex_hull_points[simplex][:, 0],
                                       convex_hull_points[simplex][:, 1],
                                       convex_hull_points[simplex][:, 2],
                                       solid_convex_hull.shape)
                solid_convex_hull[rr, cc, zz] = 1

            return solid_convex_hull

        def is_ellipsoid_full(image, c, a, b, c_axis, tolerance=0.05):
            """检查以c为中心，半轴长度为(a, b, c_axis)的椭球体是否包含最多tolerance比例的背景"""
            x, y, z = np.indices(image.shape)
            ellipsoid = ((x - c[0])**2 / a**2 + (y - c[1])**2 / b**2 + (z - c[2])**2 / c_axis**2) <= 1
            # 获取椭球体内的所有点
            ellipsoid_points = image[ellipsoid]
            print(ellipsoid_points.size, c, a, b, c_axis)
            # 计算前景的比例
            foreground_ratio = np.sum(ellipsoid_points == 1) / ellipsoid_points.size
            # 检查前景比例是否符合容忍度要求
            return foreground_ratio >= (1 - tolerance)

        def find_soma_region_in_roi(image, center, block_size=50):
            xy_resolution = float(self.meta_info['xy_resolution'].values[0])
            z_resolution = float(self.meta_info['z_resolution'].values[0])

            if(center == (0, 0, 0)):
                center = (int(image.shape[0] / 2), int(image.shape[1] / 2), int(image.shape[2] / 2))
                block_size = block_size * 2

            background = np.zeros_like(image)
            z_start, z_end = max(center[0] - int(block_size / (z_resolution / 1000)), 0), min(center[0] + int(block_size / (z_resolution / 1000)), image.shape[0])
            y_start, y_end = max(center[1] - int(block_size / (xy_resolution / 1000)), 0), min(center[1] + int(block_size / (xy_resolution / 1000)), image.shape[1])
            x_start, x_end = max(center[2] - int(block_size / (xy_resolution / 1000)), 0), min(center[2] + int(block_size / (xy_resolution / 1000)), image.shape[2])
            image = image[z_start:z_end, y_start:y_end, x_start:x_end]



            somaregionfinder = SmartSomaRegionFinder_v2()
            soma_region = somaregionfinder.test_soma_detectison(image, float(xy_resolution/1000))

            full_ellipsoid_image = np.zeros_like(background)
            if(soma_region is None):
                return None
            full_ellipsoid_image[z_start:z_end, y_start:y_end, x_start:x_end] = soma_region
            return full_ellipsoid_image


        def find_max_ellipsoid(image, center, min_radius=(5, 5, 5), tolerance=0.05, block_size=25):
            background = np.zeros_like(image)
            z_start, z_end = max(center[0] - block_size, 0), min(center[0] + block_size, image.shape[0])
            y_start, y_end = max(center[1] - block_size, 0), min(center[1] + block_size, image.shape[1])
            x_start, x_end = max(center[2] - block_size, 0), min(center[2] + block_size, image.shape[2])
            image = image[z_start:z_end, y_start:y_end, x_start:x_end]

            xy_resolution = self.meta_info['xy_resolution'].values[0]

            # 初始化半轴的最小值
            a, b, c_axis = min_radius
            b = int(b * 1000 / xy_resolution)
            c_axis = int(c_axis * 1000 / xy_resolution)
            center = (center[0] - z_start, center[1] - y_start, center[2] - x_start)


            # 逐步增加半轴的大小，直到不能满足全是前景（容忍度）
            while True:
                # 检查当前椭球体是否符合前景和背景容忍度要求
                if not is_ellipsoid_full(image, center, a, b, c_axis, tolerance):
                    break

                # 增加每个方向上的半轴长度
                a += 1
                b += 1
                c_axis += 1

            while True:
                a += 1
                if not is_ellipsoid_full(image, center, a, b, c_axis, tolerance):
                    break
            while True:
                b += 1
                if not is_ellipsoid_full(image, center, a, b, c_axis, tolerance):
                    break
            while True:
                c_axis += 1
                if not is_ellipsoid_full(image, center, a, b, c_axis, tolerance):
                    break

            # 生成最大椭球体的二值化图像
            x, y, z = np.indices(image.shape)
            ellipsoid = ((x - center[0])**2 / a**2 + (y - center[1])**2 / b**2 + (z - center[2])**2 / c_axis**2) <= 1
            ellipsoid_image = np.zeros_like(image)
            ellipsoid_image[ellipsoid] = 1

            # 将椭球体放回原始图像的坐标系中
            full_ellipsoid_image = np.zeros_like(background)
            full_ellipsoid_image[z_start:z_end, y_start:y_end, x_start:x_end] = ellipsoid_image
            return full_ellipsoid_image


        if(not input_files):
            return
        seg_file = input_files[0]
        soma_region_file = output_files[0]
        soma_marker_file = output_files[1]

        if ((not os.path.exists(seg_file)) or os.path.exists(soma_region_file)):
            return


        soma_x, soma_y, soma_z = self.meta_info['soma_x'].values[0], self.meta_info['soma_y'].values[0], self.meta_info['soma_z'].values[0]
        soma_x, soma_y, soma_z = int(float(soma_x)), int(float(soma_y)), int(float(soma_z))
        seg = tifffile.imread(seg_file)
        # somaregionfinder = SmartSomaRegionFinder()
        # soma_region = somaregionfinder.test_soma_detectison([seg_file, df, soma_region_file])


        seg = np.flip(seg, axis=1)
        # soma_region = find_max_ellipsoid(seg, (soma_z, soma_y, soma_x))
        soma_region = find_soma_region_in_roi(seg, (soma_z, soma_y, soma_x))
        if(soma_region is None):
            return

        # marker_str = f"{float(soma_x)}, {float(soma_y)}, {float(soma_z)}, 1, 1, , , 255,0,0"
        centroid = compute_centroid(soma_region)
        calc_soma_x, calc_soma_y, calc_soma_z, calc_soma_r = centroid[2], centroid[1], centroid[0], 1
        calc_soma_y = soma_region.shape[1] - calc_soma_y
        marker_str = f"{calc_soma_x}, {calc_soma_y}, {calc_soma_z}, {calc_soma_r}, 1, , , 255,0,0"

        with open(soma_marker_file, 'w') as f:
            f.write(marker_str)

        soma_region = np.where(soma_region > 0, 1, 0).astype("uint8")
        soma_region = np.flip(soma_region, axis=1)

        tifffile.imwrite(soma_region_file, soma_region * 255, compression='zlib')


    def skel_with_soma(self, input_files, output_files):
        if(not input_files):
            return
        skel_file = input_files[0]
        soma_region_file = input_files[1]
        skel_with_soma_file = output_files[0]

        if ((not os.path.exists(skel_file)) or (not os.path.exists(soma_region_file)) or os.path.exists(skel_with_soma_file)):
            return

        skel = tifffile.imread(skel_file).astype("uint8")
        soma = tifffile.imread(soma_region_file).astype("uint8")
        skelwithsoma = (np.logical_or(skel, soma))
        # dilate
        # skelwithsoma = binary_dilation(skelwithsoma, iterations=1).astype("uint8")

        # cross_structure = np.zeros((3, 3, 3))
        # cross_structure[1, :, 1] = 1
        # cross_structure[:, 1, 1] = 1
        # cross_structure[1, 1, :] = 1
        # skelwithsoma = binary_dilation(skelwithsoma, iterations=1, structure=cross_structure).astype("uint8")


        # normalize to [0, 255]
        skelwithsoma = (skelwithsoma * 255).astype("uint8")
        tifffile.imwrite(skel_with_soma_file, skelwithsoma, compression='zlib')
        # tifffile.imwrite(skel_with_soma_file, skelwithsoma)

    def trace_gtree(self, input_files, output_files):
        v3d_v3_path = r"/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin/vaa3d"
        def process_path(path):
            return path.replace('\\', '/')
        if(not input_files):
            return
        img_file = input_files[0]
        swc_file = output_files[0]

        result_name = img_file + "_NeuroGPSTree.swc"
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_v3_path} -x NeuroGPSTree -f tracing_func -i {img_file} -p 1 1 1 10'

        env = os.environ.copy()
        ld_library_path = v3d_v3_path[:-6]  # '/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin'
        env['LD_LIBRARY_PATH'] = f"{ld_library_path}:{env.get('LD_LIBRARY_PATH', '')}"
        # os.system(f"{cmd} env {env}")
        print(cmd)
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

        if (os.path.exists(result_name)):
            os.rename(result_name, swc_file)





    def trace_app2_with_soma(self, input_files, output_files):
        def process_path(path):
            return path.replace('\\', '/')
        if(not input_files):
            return
        img_file = input_files[0]
        # somamarker_file = input_files[1]
        somamarker_file = 'NULL'
        swc_file = output_files[0]
        if(not os.path.exists(img_file) or os.path.exists(swc_file)):
            return
        ini_swc_path = img_file.replace('.tif', '.tif_ini.swc')
        '''
            **** Usage of APP2 ****
            vaa3d -x plugin_name -f app2 -i <inimg_file> -o <outswc_file> -p [<inmarker_file> [<channel> [<bkg_thresh> 
            [<b_256cube> [<b_RadiusFrom2D> [<is_gsdt> [<is_gap> [<length_thresh> [is_resample][is_brightfield][is_high_intensity]]]]]]]]]
            inimg_file          Should be 8/16/32bit image
            inmarker_file       If no input marker file, please set this para to NULL and it will detect soma automatically.
                                When the file is set, then the first marker is used as root/soma.
            channel             Data channel for tracing. Start from 0 (default 0).
            bkg_thresh          Default 10 (is specified as AUTO then auto-thresolding)
            b_256cube           If trace in a auto-downsampled volume (1 for yes, and 0 for no. Default 1.)
            b_RadiusFrom2D      If estimate the radius of each reconstruction node from 2D plane only (1 for yes as many 
            times the data is anisotropic, and 0 for no. Default 1 which which uses 2D estimation.)
            is_gsdt             If use gray-scale distance transform (1 for yes and 0 for no. Default 0.)
                           If allow gap (1 for yes and 0 for no. Default 0.)
            length_thresh       Default 5
            is_resample         If allow resample (1 for yes and 0 for no. Default 1.)
            is_brightfield      If the signals are dark instead of bright (1 for yes and 0 for no. Default 0.)
            is_high_intensity   If the image has high intensity background (1 for yes and 0 for no. Default 0.)
            outswc_file         If not be specified, will be named automatically based on the input image file name.
        '''

        resample = 1
        gsdt = 1
        b_RadiusFrom2D = 1
        b_256cube = 0 # 图像重采样到256立方体

        try:
            if (sys.platform == "linux"):
                cmd = f'xvfb-run -a -s "-screen 0 640x480x16" "{v3d_path}" -x vn2 -f app2 -i "{img_file}" -o "{swc_file}" -p "{somamarker_file}" 0 10 {b_256cube} {b_RadiusFrom2D} {gsdt} 1 {resample} 0 0'
                cmd = process_path(cmd)
                subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
            else:
                pass
        except:
            pass

        if (os.path.exists(ini_swc_path)):
            os.remove(ini_swc_path)

    def prune_fiber_in_soma(self, point_l, soma_region):
        edge_p_list = []
        x_limit, y_limit, z_limit = soma_region.shape[2], soma_region.shape[1], soma_region.shape[0]

        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            x, y, z = p.x, p.y, p.z
            y = soma_region.shape[1] - y

            x = min(int(x), x_limit - 1)
            y = min(int(y), y_limit - 1)
            z = min(int(z), z_limit - 1)

            if (soma_region[int(z), int(y), int(x)]):
                edge_p_list.append(p)

        for p in edge_p_list:
            if (len(p.s) == 0):
                temp_p = point_l.p[p.n]
                while (True):
                    if (temp_p.n == 1): break
                    if (temp_p.pruned == True): break
                    if (not len(temp_p.s) == 1): break
                    point_l.p[temp_p.n].pruned = True
                    temp_p = point_l.p[temp_p.p]
        for p in point_l.p:
            for s in p.s:
                if (point_l.p[s].pruned == True):
                    p.s.remove(s)

        return point_l

    def connect_to_soma_file(self, input_files, output_files):
        if (not input_files):
            return
        swc_file = input_files[0]
        soma_refion_file = input_files[1]
        conn_swc_file = output_files[0]

        if ((not os.path.exists(swc_file)) or (not os.path.exists(soma_refion_file)) or os.path.exists(conn_swc_file)):
            return

        soma_region = tifffile.imread(soma_refion_file).astype("uint8")
        # soma_region = get_main_soma_region_in_msoma_from_gsdt(soma_region,,
        soma_region = binary_dilation(soma_region, iterations=4).astype("uint8")
        x_limit, y_limit, z_limit = soma_region.shape[2], soma_region.shape[1], soma_region.shape[0]

        point_l = read_swc(swc_file)

        labeled_img, num_objects = ndimage.label(soma_region)
        if (len(point_l.p) <= 1):
            return
        for obj_id in range(1, num_objects + 1):
            obj_img = np.where(labeled_img == obj_id, 1, 0)
            x, y, z = point_l.p[1].x, point_l.p[1].y, point_l.p[1].z
            if (obj_img[int(z), int(y), int(x)]):
                soma_region = obj_img
                del obj_img
                break
            del obj_img
        # tifffile.imwrite(os.path.join(conn_folder, file_name+"1.tif"), soma_region.astype("uint8")*255, compression='zlib')
        labeled_img, num_objects = ndimage.label(soma_region)
        if (num_objects > 1):
            # soma_region = dusting(soma_region)
            write_swc(conn_swc_file, point_l)
            del soma_region, point_l
            return

        point_l = self.prune_fiber_in_soma(point_l, soma_region)

        # strict strategy
        edge_p_list = []
        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            x, y, z = p.x, p.y, p.z
            y = soma_region.shape[1] - y

            x = min(int(x), x_limit - 1)
            y = min(int(y), y_limit - 1)
            z = min(int(z), z_limit - 1)

            if (soma_region[int(z), int(y), int(x)]):
                edge_p_list.append(p)

        for p in edge_p_list:
            temp_p = point_l.p[p.p]
            while (True):
                if (temp_p.n == 1): break
                if (temp_p.pruned == True): break
                point_l.p[temp_p.n].pruned = True
                temp_p = point_l.p[temp_p.p]

        for p in edge_p_list:
            if (point_l.p[p.n].pruned == False):
                if (not len(point_l.p[p.n].s)):
                    point_l.p[p.n].pruned = True
                else:
                    point_l.p[p.n].p = 1
                    point_l.p[1].s.append(p.n)
            else:
                for s in point_l.p[p.n].s:
                    point_l.p[s].p = 1
                    point_l.p[1].s.append(s)

        # Conservative strategy
        # for s in point_l.p[1].s:
        #     temp_p = point_l.p[s]
        #     x, y, z = temp_p.x, temp_p.y, temp_p.z
        #     y = soma_region.shape[1] - y
        #     x = min(int(x), x_limit - 1)
        #     y = min(int(y), y_limit - 1)
        #     z = min(int(z), z_limit - 1)
        #
        #     if(not soma_region[int(z), int(y), int(x)]):
        #         continue
        #
        #     for s2 in point_l.p[s].s:
        #         point_l.p[s2].p = 1
        #         point_l.p[1].s.append(s2)
        #
        #     point_l.p[1].s.remove(s)
        #     point_l.p[s].pruned = True
        # print(conn_path)
        if (os.path.exists(conn_swc_file)):
            os.remove(conn_swc_file)
        write_swc(conn_swc_file, point_l)
        # print(len(point_l.p))
        del soma_region, point_l

    def rescale_xy_resolution(self, input_files, output_files):
        if(not input_files):
            return

        swc_file = input_files[0]
        unified_swc_file = output_files[0]

        if(not os.path.exists(swc_file) or os.path.exists(unified_swc_file)):
            return

        # xy_resolution = float(self.meta_info['xy_resolution'].values[0])
        # print(xy_resolution)
        # with open(swc_file, 'r') as f:
        #     lines = f.readlines()
        # # print(len(lines))
        # with open(unified_swc_file, 'w') as f:
        #     result_lines = []
        #     for line in lines:
        #         if line.startswith("#"):
        #             result_lines.append(line)
        #         else:
        #             line = line.strip().split()
        #             line[2] = str(float(line[2]) * xy_resolution / 1000)
        #             line[3] = str(float(line[3]) * xy_resolution / 1000)
        #             result_lines.append(" ".join(line) + "\n")
        #     f.writelines(result_lines)

        xy_resolution, z_resolution = float(self.meta_info['xy_resolution'].values[0]), float(self.meta_info['z_resolution'].values[0])
        swc_point_list = pd.read_csv(swc_file, delim_whitespace=True, comment='#', header=None,
                                        names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'])
        swc_point_list['x'] = swc_point_list['x'] * xy_resolution / 1000
        swc_point_list['y'] = swc_point_list['y'] * xy_resolution / 1000
        swc_point_list['z'] = swc_point_list['z'] * z_resolution / 1000

        swc_point_list.to_csv(unified_swc_file, sep=' ', header=False, index=False)

    def get_estimated_radius(self, input_files, output_files):
        def v3d_get_radius(img_path, swc_path, out_path):
            with tempfile.TemporaryDirectory() as temp_dir:
                # 获取文件名
                img_filename = os.path.basename(img_path).split('_')[0] + '.tif'
                swc_filename = os.path.basename(swc_path).split('_')[0] + '.swc'
                output_filename = os.path.basename(out_path).split('_')[0] + '.swc'

                # 设置缓存文件路径
                img_cache_path = os.path.join(temp_dir, img_filename)
                swc_cache_path = os.path.join(temp_dir, swc_filename)
                out_cache_path = os.path.join(temp_dir, output_filename)

                # 将文件复制到缓存路径
                shutil.copy(img_path, img_cache_path)
                shutil.copy(swc_path, swc_cache_path)

                # 设置命令字符串
                radius2d = 1
                cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x neuron_radius -f neuron_radius -i {img_cache_path} {swc_cache_path} -o {out_cache_path} -p 10 {radius2d}'
                cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')

                # 执行命令
                # print(f"Running command: {cmd_str}")
                subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)

                # 将结果从临时路径复制到实际输出路径
                shutil.copy(out_cache_path, out_path)

        def load_swc_to_undirected_graph(swc_file_path):
            """从SWC文件加载数据，构建无向图，并记录每个节点的parent信息"""
            df = pd.read_csv(swc_file_path, delim_whitespace=True, comment='#', header=None,
                             names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'])
            G = nx.Graph()

            for _, row in df.iterrows():
                # 添加节点，同时记录parent信息
                G.add_node(row['id'], pos=(row['x'], row['y'], row['z']), radius=row['radius'], type=row['type'],
                           parent=row['parent'])
                if row['parent'] != -1:
                    G.add_edge(row['parent'], row['id'])

            return G

        def find_nearest_node(G, target_pos):
            """ 在图中找到与给定坐标最近的节点 """
            nearest_node = None
            min_distance = float('inf')

            for node in G.nodes(data=True):
                pos = node[1]['pos']
                distance = np.linalg.norm(np.array(pos) - np.array(target_pos))
                if distance < min_distance:
                    nearest_node = node[0]
                    min_distance = distance

            return nearest_node

        def export_to_swc_dfs(G, root_pos, output_filename):
            if(os.path.exists(output_filename)):
                os.remove(output_filename)

            start_node = find_nearest_node(G, root_pos)

            # 调整根节点
            potential_root = max(G.nodes, key=lambda x: G.degree(x))
            potential_root_degree = G.degree(potential_root)
            potential_root_list = [node for node in G.nodes if G.degree(node) == potential_root_degree]
            for node in potential_root_list:
                if G.degree(node) > 4 and len(potential_root_list) == 1:  # 这个点的度数大于4
                    start_node = node
                elif (nx.shortest_path_length(G, start_node, node) < 3):
                    start_node = node
                elif (np.linalg.norm(np.array(G.nodes[node]['pos']) - np.array(root_pos)) < 10):
                    start_node = node

            # 打开文件进行写入
            with open(output_filename, 'w') as f:
                # 写入SWC文件的头部注释
                f.write("# SWC file generated from DFS traversal\n")
                f.write("# Columns: id type x y z radius parent\n")

                # 用于存储节点的新编号和访问状态
                new_id = 1
                visited = set()
                stack = [(start_node, -1)]  # (current_node, parent_id_in_new_swc)

                while stack:
                    node, parent_id = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        node_data = G.nodes[node]
                        pos = node_data['pos']
                        radius = node_data['radius']
                        if (parent_id == -1):
                            node_type = 1
                        else:
                            node_type = 3

                        # 写入当前节点数据
                        f.write(f"{new_id} {node_type} {pos[0]} {pos[1]} {pos[2]} {radius} {parent_id}\n")

                        # 更新父节点ID为当前节点的新ID
                        current_parent_id = new_id
                        new_id += 1

                        # 将所有未访问的邻接节点添加到栈中
                        for neighbor in G.neighbors(node):
                            if neighbor not in visited:
                                stack.append((neighbor, current_parent_id))

        def calc_node_dist(G, node1, node2):
            pos1 = np.array(G.nodes[node1]['pos'])
            pos2 = np.array(G.nodes[node2]['pos'])
            return np.linalg.norm(pos1 - pos2)

        def gaussian_smoothing_radius_tree(G, sigma=1.0):
            smoothed_values = {}
            soma_r = G.nodes[1]['radius']
            for node in G.nodes:
                neighbors = list(G.neighbors(node))
                weights = []
                values = []
                for neighbor in neighbors:
                    distance = calc_node_dist(G, node, neighbor)
                    weight = np.exp(- (distance ** 2) / (2 * sigma ** 2))
                    weights.append(weight)
                    values.append(G.nodes[neighbor]['radius'])
                # 自身的权重
                self_weight = np.exp(0)
                total_weight = self_weight + sum(weights)
                weighted_sum = G.nodes[node]['radius'] * self_weight + sum(w * v for w, v in zip(weights, values))
                smoothed_values[node] = weighted_sum / total_weight
            nx.set_node_attributes(G, smoothed_values, 'radius')
            G.nodes[1]['radius'] = soma_r
            return G

        def smoothing_swc_file(swc_file_path, output_filename):
            G = load_swc_to_undirected_graph(swc_file_path)
            G = gaussian_smoothing_radius_tree(G)
            root_pos = G.nodes[1]['pos']
            # print(root_pos)
            export_to_swc_dfs(G, root_pos, output_filename)


        if(not input_files):
            return

        swc_file = input_files[0]
        seg_file = input_files[1]
        radius_swc_file = output_files[0]

        if(not os.path.exists(swc_file) or not os.path.exists(seg_file) or os.path.exists(radius_swc_file)):
            return

        seg = tifffile.imread(seg_file).astype("uint8")
        seg = np.flip(seg, axis=1)
        origin_shape = seg.shape
        xy_resolution = self.meta_info['xy_resolution'].values[0]
        img_shape = [int(origin_shape[0]), int(origin_shape[1] * xy_resolution / 1000),
                     int(origin_shape[2] * xy_resolution / 1000)]
        seg = skimage.transform.resize(seg, img_shape, order=0, anti_aliasing=False)
        # print(img_shape)
        seg = np.where(seg > 0, 255, 0).astype("uint8")
        temp_img_file = radius_swc_file.replace('.swc', '_temp.tif')
        tifffile.imwrite(temp_img_file, seg)

        v3d_get_radius(temp_img_file, swc_file, radius_swc_file)
        try:
            smoothing_swc_file(radius_swc_file, radius_swc_file)
        except:
            print(f"swc radius estimate error: {radius_swc_file}")
            if(os.path.exists(radius_swc_file)):
                os.remove(radius_swc_file)
        # print(temp_img_file)
        os.remove(temp_img_file)

def process_pipeline(work_dir, file_name, meta_info):




    pipeline = AutoTracePipeline(work_dir, file_name, meta_info)
    # print("Processing", file_name, "...")
    # print("Meta info:", meta_info)
    pipeline.run()
    #
    # pr.disable()
    # pr.dump_stats('profile.prof')
    # p = pstats.Stats('profile.prof')
    # p.sort_stats('cumulative').print_stats(10)

    return file_name


    # try:
    #     pipeline = AutoTracePipeline(work_dir, file_name, meta_info)
    #     pipeline.run()  # 如果需要执行任务，可以取消注释这行
    #     return file_name
    # except Exception as e:
    #     print(f"Error processing {file_name}: {e}")
    #     return None