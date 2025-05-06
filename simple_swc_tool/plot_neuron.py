import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tifffile
from tqdm import tqdm
from joblib import Parallel, delayed

# dpi = 300
plt.rcParams['figure.dpi'] = 300

def encoding_projection_direction(projection_direction):
    if projection_direction == 'xy':
        projection_axes = 0
    elif projection_direction == 'xz':
        projection_axes = 1
    elif projection_direction == 'yz':
        projection_axes = 2
    return projection_axes

def load_swc_file(swc_file):
    swc = pd.read_csv(swc_file, sep=' ', header=None, comment='#')
    swc.columns = ['n', 'type', 'x', 'y', 'z', 'r', 'parent']
    return swc

def plot_img_on_fig(fig, gray_img, projection_direction='xy', alpha=0):
    projection_axes = encoding_projection_direction(projection_direction)
    gray_img = np.max(gray_img, axis=projection_axes)
    # resize
    gray_img = cv2.resize(gray_img, (fig.shape[1], fig.shape[0]))
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # print(gray_img.shape, fig.shape)

    fig = cv2.addWeighted(fig, alpha, gray_img, 1 - alpha, 0)
    return fig

def plot_swc_on_fig(fig, swc_file, plot_mode="sphere", projection_direction='xy', line_color=(255, 0, 0), line_thickness=1, soma_color=(0, 0, 255), soma_thickness=3):
    projection_axes = encoding_projection_direction(projection_direction)

    swc_points = load_swc_file(swc_file)

    for swc_point_id in range(len(swc_points)):
        if(plot_mode == 'sphere'):
            if (projection_axes == 0):
                # cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
            elif (projection_axes == 1):
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].z)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
            elif (projection_axes == 2):
                cv2.circle(fig, (int(swc_points.iloc[swc_point_id].y), int(swc_points.iloc[swc_point_id].z)), int(swc_points.iloc[swc_point_id].r), line_color, -1)
        # elif(plot_mode=="line"):
        #     if (projection_axes == 0):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].y)), 1, line_color, -1)
        #     elif (projection_axes == 1):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].x), int(swc_points.iloc[swc_point_id].z)), 1, line_color, -1)
        #     elif (projection_axes == 2):
        #         cv2.circle(fig, (int(swc_points.iloc[swc_point_id].y), int(swc_points.iloc[swc_point_id].z)), 1, line_color, -1)

        if swc_points.iloc[swc_point_id].parent == -1:
            continue
        swc_point = swc_points.iloc[swc_point_id]
        parent_point = swc_points[swc_points['n'] == swc_point['parent']].iloc[0]
        # print(swc_point, parent_point)

        nx, ny, nz = swc_point.x, swc_point.y, swc_point.z
        px, py, pz = parent_point.x, parent_point.y, parent_point.z

        if (projection_axes == 0):
            cv2.line(fig, (int(nx), int(ny)), (int(px), int(py),), line_color, line_thickness)
        elif (projection_axes == 1):
            cv2.line(fig, (int(nx), int(nz)), (int(px), int(pz),), line_color, line_thickness)
        elif (projection_axes == 2):
            cv2.line(fig, (int(ny), int(nz)), (int(py), int(pz),), line_color, line_thickness)


    if (projection_axes == 0):
        cv2.circle(fig, (int(swc_points.iloc[0].x), int(swc_points.iloc[0].y)), soma_thickness, soma_color, -1)
    elif (projection_axes == 1):
        cv2.circle(fig, (int(swc_points.iloc[0].x), int(swc_points.iloc[0].z)), soma_thickness, soma_color, -1)
    elif (projection_axes == 2):
        cv2.circle(fig, (int(swc_points.iloc[0].y), int(swc_points.iloc[0].z)), soma_thickness, soma_color, -1)

    return fig

def plot_markers_on_fig(fig, markers_file, projection_direction='xy', markers_color=(0, 255, 0), thickness=3, marker_type='rectangle'):
    markers = pd.read_csv(markers_file, sep=',', header=None, comment='#')
    ##x,y,z,radius,shape,name,comment,color_r,color_g,color_b
    markers.columns = ['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g', 'color_b']
    projection_axes = encoding_projection_direction(projection_direction)

    for marker_id in range(len(markers)):
        marker = markers.iloc[marker_id]
        if marker_type == 'circle':
            if (projection_axes == 0):
                cv2.circle(fig, (int(marker.x), int(marker.y)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
            elif (projection_axes == 1):
                cv2.circle(fig, (int(marker.x), int(marker.z)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
            elif (projection_axes == 2):
                cv2.circle(fig, (int(marker.y), int(marker.z)), int(marker.radius), (marker.color_b, marker.color_g, marker.color_r), thickness)
        elif marker_type == 'rectangle':
            if (projection_axes == 0):
                cv2.rectangle(fig, (int(marker.x - marker.radius), int(marker.y - marker.radius)), (int(marker.x + marker.radius), int(marker.y + marker.radius)), markers_color, thickness)
            elif (projection_axes == 1):
                cv2.rectangle(fig, (int(marker.x - marker.radius), int(marker.z - marker.radius)), (int(marker.x + marker.radius), int(marker.z + marker.radius)), markers_color, thickness)
            elif (projection_axes == 2):
                cv2.rectangle(fig, (int(marker.y - marker.radius), int(marker.z - marker.radius)), (int(marker.y + marker.radius), int(marker.z + marker.radius)), markers_color, thickness)

    return fig

if __name__ == '__main__':
    swc_file = '/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/9_soma_g_cut_swc/14953.swc'
    save_file = '/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/9_soma_g_cut_swc/14953.png'

    img_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/image"
    swc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/7_scaled_1um_swc"
    # pruned_swc_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/9_my_cut_swc"
    marker_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/mapped_markers"
    mip_dir = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/recon_temp/e_cut_result"
    os.makedirs(mip_dir, exist_ok=True)

    meta_info_file = "/data2/kfchen/tracing_ws/human_neuron_recon_50k/meta_info_0205.xlsx"
    meta_info = pd.read_excel(meta_info_file)

    def current_task(img_file):
        neuron_id = int(img_file.split('_')[1])
        swc_file = os.path.join(swc_dir, img_file.replace('_0000.tif', '.swc').replace("image_", ""))
        # pruned_swc_file = os.path.join(pruned_swc_dir, img_file.replace('_0000.tif', '.swc').replace("image_", ""))
        marker_file = os.path.join(marker_dir, img_file.replace('_0000.tif', '.txt').replace("image_", ""))
        if (not os.path.exists(swc_file) or not os.path.exists(marker_file)):
            return
        save_file = os.path.join(mip_dir, str(neuron_id) + ".png")
        if (os.path.exists(save_file)):
            return


        xy_resolution = meta_info[meta_info['cell_id'] == neuron_id]['xy_resolution'].values[0]
        img = tifffile.imread(os.path.join(img_dir, img_file))
        img_mip = np.max(img, axis=0)

        bkg_shape = (int(img_mip.shape[0] * xy_resolution / 1000), int(img_mip.shape[1] * xy_resolution / 1000))
        background = np.ones(bkg_shape).astype(np.uint8) * 255
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

        background = plot_img_on_fig(background, img)
        background = plot_swc_on_fig(background, swc_file, line_color=(255, 0, 0))
        # background = plot_swc_on_fig(background, pruned_swc_file, line_color=(0, 0, 255))
        background = plot_markers_on_fig(background, marker_file)

        plt.imshow(background)
        plt.savefig(save_file)
        plt.close()

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    # for img_file in tqdm(img_files):
    #     current_task()
    Parallel(n_jobs=8)(delayed(current_task)(img_file) for img_file in tqdm(img_files))


