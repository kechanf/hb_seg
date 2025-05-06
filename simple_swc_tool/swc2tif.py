import os
import subprocess
import platform


def swc2img(swc_path, img_shape, out_path=None, v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i <input.swc> [-p <sz0> <sz1> <sz2>] [-o <output_image.raw>]\n"
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_filter -i <input.tif> <input.swc> [-o <output_image.raw>]\n"

    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_path} /x swc_to_maskimage_sphere_unit /f swc_to_maskimage /i {swc_path} '
            f'/p {img_shape[2]} {img_shape[1]} {img_shape[0]} /o {out_path}',
            stdout=subprocess.DEVNULL)  # 全路径
    else:
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i {swc_path} ' \
                  f'-p {img_shape[2]} {img_shape[1]} {img_shape[0]} -o {out_path}'
        cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
        # print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
    return out_path