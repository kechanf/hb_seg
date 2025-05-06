import os
import subprocess
import sys
import uuid
import threading
import time


def process_path(pstr):
    return pstr.replace('(', '\(').replace(')', '\)')

def trace_init(method, img_file, somamarker_file, out_swc_file):
    if (os.path.exists(out_swc_file)):
        print(f"{method} result exists: {out_swc_file}")
        return out_swc_file, somamarker_file
    if(out_swc_file==None):
        out_swc_file = os.path.join(os.path.dirname(img_file), str(method),
                                    os.path.basename(img_file).replace(".tif", ".swc"))
    if not os.path.exists(os.path.dirname(out_swc_file)):
        os.makedirs(os.path.dirname(out_swc_file), exist_ok=True)
    if (somamarker_file==None or not os.path.exists(somamarker_file)):
        somamarker_file = "NULL"

    return out_swc_file, somamarker_file

def APP1_trace_file(img_file, out_swc_file=None, v3d_path=None, somamarker_file=None):
    """
    **** Usage of APP1 ****
    vaa3d -x plugin_name -f app1 -i <inimg_file> -p [<inmarker_file> [<channel> [<bkg_thresh> [<b_256cube> ]]]]
    inimg_file       Should be 8/16/32bit image
    inmarker_file    If no input marker file, please set this para to NULL and it will detect soma automatically.
                     When the file is set, then the first marker is used as root/soma.
    channel          Data channel for tracing. Start from 0 (default 0).
    bkg_thresh       Default AUTO (AUTO is for auto-thresholding), otherwise the threshold specified by a user will be used.
    b_256cube        If trace in a auto-downsampled volume (1 for yes, and 0 for no. Default 1.)
    outswc_file      If not be specified, will be named automatically based on the input image file name.
    """
    out_swc_file, somamarker_file = trace_init("app1", img_file, somamarker_file, out_swc_file)
    if (os.path.exists(out_swc_file)):return

    if (sys.platform == "linux"):
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x vn2 -f app1 -i {img_file} -o {out_swc_file} -p {somamarker_file} 0 AUTO 0'
        cmd = process_path(cmd)
        # print(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
    else:
        cmd = f'{v3d_path} /x D:/tracing_ws/vn2.dll /f app1 /i {img_file} /o {out_swc_file} /p {somamarker_file} 0 10 1'
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if(os.path.exists(out_swc_file)):
        # print(f"APP1 tracing done: {out_swc_file}")
        pass
    else:
        print(f"APP1 tracing failed: {out_swc_file}")
        print(cmd)

def APP2_trace_file(img_file, out_swc_file=None, v3d_path=None, somamarker_file=None, resample=1, gsdt=1, b_RadiusFrom2D=1):
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
    out_swc_file, somamarker_file = trace_init("app2", img_file, somamarker_file, out_swc_file)
    if (os.path.exists(out_swc_file)): return

    if (sys.platform == "linux"):
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x vn2 -f app2 -i {img_file} -o {out_swc_file} -p {somamarker_file} 0 10 1 {b_RadiusFrom2D} {gsdt} 1 {resample} 0 0'
        cmd = process_path(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
    else:
        vn2_path = r"E:/tracing_ws/vn2.dll"
        cmd = f'{v3d_path} /x {vn2_path} /f app2 /i {img_file} /o {out_swc_file} /p {somamarker_file} 0 10 1 1 {gsdt} 1 5 {resample} 0 0'
        # print(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    print(f"APP2 tracing done: {out_swc_file}")

def Advantra_trace_file(img_file, out_swc_file=None, v3d_path=None):
    '''
    printf("**** Usage of Advantra tracing **** \n");
        printf("vaa3d -x Advantra -f advantra_func -i <inimg_file> -p <scal bratio znccTh Ndir angSig Ni Ns zDist>\n");
        printf("inimg_file          The input image\n");
        printf("scal                Scale (5, 20] pix.\n"); default 10
        printf("bratio              Background ratio (0, 1].\n"); default 0.5
    //    printf("perc                Percentile [50, 100].\n");
        printf("znccTh              Correlation threshold [0.5, 1.0).\n"); default 0.75
        printf("Ndir                nr. directions [5, 20].\n"); default 10
        printf("angSig              Angular sigma [20,90] degs.\n"); default 60
        printf("Ni                  nr. iterations [2, 50].\n"); default 5
        printf("Ns                  nr. states [1, 20].\n"); default 5
        printf("zDist               z layer dist [1, 4] pix.\n"); default 1
        printf("outswc_file         Will be named automatically based on the input image file name, so you don't have to specify it.\n\n");
    }
    '''
    out_swc_file, _ = trace_init("Advantra", img_file, out_swc_file=out_swc_file, somamarker_file=None)
    if (os.path.exists(out_swc_file)): return

    if (sys.platform == "linux"):
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x Advantra -f advantra_func -i {img_file} -p 10 0.5 0.75 10 60 5 5 1 -o {out_swc_file}'
        cmd = process_path(cmd)
        # print(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
    else:
        pass

    # rename file
    result_file = img_file + "_Advantra.swc"

    if(os.path.exists(result_file)):
        os.rename(result_file, out_swc_file)
        # print(f"Advantra tracing done: {out_swc_file}")
    else:
        print(f"Advantra tracing failed: {out_swc_file}")

def Meanshift_trace_file(img_file):
    '''
    printHelp();
    printf("**** Usage of meanshift_spanning tracing **** \n");
    printf("vaa3d -x BJUT_meanshift -f meanshift -i <inimg_file> -p <channel> <prim_distance> <threshold> <percentage> \n");
    printf("inimg_file       The input image\n");
    printf("channel       image channel, default 1.\n");
    printf("prim_distance       the distance to delete the covered nodes.\n");
    printf("threshold  the pixal threshold to determine noisy.\n");
    printf("percentage          same effect with threshold.\n");
    :param img_file:
    :return:
    '''

    if (sys.platform == "linux"):
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x BJUT_meanshift -f meanshift -i {img_file} -p 1 3 10 0.6'
        cmd = process_path(cmd)
        print(cmd)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
    else:
        pass

def Fastmarching_spanningtree_trace_file(img_file, out_swc_file=None, v3d_path=None):
    '''
    printf("**** Usage of fastmarching_spanningtree tracing **** \n");
    printf("vaa3d -x fastmarching_spanningtree -f tracing_func -i <inimg_file> -p <channel> <other parameters>\n");
    printf("inimg_file       The input image\n");
    printf("channel          Data channel for tracing. Start from 1 (default 1).\n");

    printf("outswc_file      Will be named automatically based on the input image file name, so you don't have to specify it.\n\n");
    '''
    if (sys.platform == "linux"):
        env = os.environ.copy()
        ld_library_path = v3d_path[:-6] # '/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin'
        env['LD_LIBRARY_PATH'] = f"{ld_library_path}:{env.get('LD_LIBRARY_PATH', '')}"

        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x fastmarching_spanningtree -f tracing_func -i {img_file} -p 1'
        cmd = process_path(cmd)
        print(cmd)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        print(result.stdout)
        print(result.stderr)
    else:
        pass

def early_stop_for_CWlab_ver1(cmd, filepath, env):

    def monitor_file_size(filepath, max_size_kb, process):
        """ Monitor the size of a file and terminate the process if it exceeds the maximum size. """
        while True:
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                if size_kb > max_size_kb:
                    # print(f"File size exceeded {max_size_kb}KB. Terminating the process.")
                    process.terminate()  # Terminate the process
                    break
            # print("checked file size")
            time.sleep(1)  # Check every second

    # Start the subprocess
    # print(f"Starting process: {cmd}")
    # subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True, env=env)
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, shell=True, env=env)

    # Start the file size monitoring thread
    max_size_kb = 1  # Maximum file size in KB
    monitor_thread = threading.Thread(target=monitor_file_size, args=(filepath, max_size_kb, process))
    monitor_thread.start()
    # print("monitor_thread started")

    # Wait for the process to complete
    process.wait()
    # print("Process completed")

    # Optionally, join the monitor thread to ensure it has completed
    monitor_thread.join()
    # print("monitor_thread joined")

def CWlab_method_v1(img_file, out_swc_file=None, v3d_path=None):
    '''
    printf("**** Usage of CWlab_method1_version1 tracing **** \n");
    printf("vaa3d -x CWlab_method1_version1 -f tracing_func -i <inimg_file> -p <channel> <other parameters>\n");
    '''
    out_swc_file, _ = trace_init("Cwlab_ver1", img_file, out_swc_file=out_swc_file, somamarker_file=None)
    if (os.path.exists(out_swc_file)): return
    result_name = img_file + "_Cwlab_ver1.swc"

    if (sys.platform == "linux"):
        # result = subprocess.run(['env'], stdout=subprocess.PIPE, text=True)
        # print(result.stdout)

        env = os.environ.copy()
        ld_library_path = v3d_path[:-6] # '/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin'
        env['LD_LIBRARY_PATH'] = f"{ld_library_path}:{env.get('LD_LIBRARY_PATH', '')}"

        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x CWlab_method1_version1 -f tracing_func -i {img_file} -p 1'
        cmd = process_path(cmd)
        # result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        early_stop_for_CWlab_ver1(cmd, result_name, env)
        # print("early stop")

    else:
        pass

    if(os.path.exists(result_name)):
        os.rename(result_name, out_swc_file)
        # print(f"CWlab_method1_version1 tracing done (early stop): {out_swc_file}")
    else:
        print(f"CWlab_method1_version1 tracing failed: {out_swc_file}")

def MOST_trace_file(img_file, out_swc_file=None, v3d_path=None):
    '''
    cout<<"Usage : v3d -x dllname -f MOST_trace -i <inimg_file> -p <ch> <th> <seed> <slip>"<<endl;
    '''
    out_swc_file, _ = trace_init("MOST", img_file, out_swc_file=out_swc_file, somamarker_file=None)
    if (os.path.exists(out_swc_file)): return
    result_name = img_file + "_MOST.swc"

    if (sys.platform == "linux"):
        env = os.environ.copy()
        ld_library_path = v3d_path[:-6]  # '/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin'
        env['LD_LIBRARY_PATH'] = f"{ld_library_path}:{env.get('LD_LIBRARY_PATH', '')}"

        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x MOST -f MOST_trace -i {img_file} -p 1'
        cmd = process_path(cmd)
        # print(cmd)
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        # print(result.stdout)
        # print(result.stderr)
    else:
        pass

    if(os.path.exists(result_name)):
        os.rename(result_name, out_swc_file)
        # print(f"MOST tracing done: {out_swc_file}")
    else:
        print(f"MOST tracing failed: {out_swc_file}")

def TReMap_trace_file(img_file, somamarker_file=None, out_swc_file=None, v3d_path=None):
    '''
    -x TReMap -f trace_mip -i '+data_filenames[0]+' -p 0 1 10 0 1 0 5'
    printf("vaa3d -x NeuronAssembler_tReMap -f trace_raw -i <inimg_file> -p <inmarker_file> <block size> <tracing_entire_image> <mip> <ch> <th> <b_256> <is_gsdt> <is_gap> <length_th>\n");
    printf("inimg_file		Should be 8 bit v3draw/raw image\n");
    printf("inmarker_file		Please specify the path of the marker file, Default value is NULL\n");
    printf("block size		Default 1024\n");
    printf("tracing_entire_image	YES:1, NO:0. Default value is 0\n");

    printf("mip			Required by the tracing algorithm. Default value is 0\n");
    printf("ch			Required by the tracing algorithm. Default value is 1\n");
    printf("th			Required by the tracing algorithm. Default value is 10\n");
    printf("b_256			Required by the tracing algorithm. Default value is 0\n");
    printf("is_gsdt			Required by the tracing algorithm. Default value is 1\n");
    printf("is_gap			Required by the tracing algorithm. Default value is 0\n");
    printf("length_th		Required by the tracing algorithm. Default value is 5\n");

    printf("outswc_file		Will be named automatically based on the input image file name, so you don't have to specify it.\n\n");
    '''
    print("??")
    out_swc_file, somamarker_file = trace_init("TReMap", img_file, out_swc_file=out_swc_file, somamarker_file=somamarker_file)
    if (os.path.exists(out_swc_file)): return

    if (sys.platform == "linux"):
        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x TReMap -f trace_mip -i {img_file} -o {out_swc_file} -p 0 1 10 0 1 0 5'
        cmd = process_path(cmd)
        print(cmd)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print(result.stdout)
        print(result.stderr)
    else:
        pass

    print(f"TReMap tracing done: {out_swc_file}")

def Mst_tracing_file(img_file, out_swc_file=None, v3d_path=None):
    '''
    -x MST_tracing -f trace_mst -i '+data_filenames[0]+' -p 1 5
    '''
    out_swc_file, _ = trace_init("MST", img_file, out_swc_file=out_swc_file, somamarker_file=None)
    if (os.path.exists(out_swc_file)): return
    result_name = img_file + "_MST_Tracing.swc"

    if (sys.platform == "linux"):
        env = os.environ.copy()
        ld_library_path = v3d_path[:-6] # '/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin'
        env['LD_LIBRARY_PATH'] = f"{ld_library_path}:{env.get('LD_LIBRARY_PATH', '')}"

        cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x MST_tracing -f trace_mst -i {img_file} -p 1 5'
        cmd = process_path(cmd)
        # print(cmd)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        # print(result.stdout)
        # print(result.stderr)
    else:
        pass

    if(os.path.exists(result_name)):
        os.rename(result_name, out_swc_file)
        # print(f"MST tracing done: {out_swc_file}")
    else:
        print(f"MST tracing failed: {out_swc_file}")

def NeuroGPSTree_trace_file(img_file, out_swc_file=None, v3d_path=None):
    # f'{self.vaa3d_path} -x NeuroGPSTree -f tracing_func -i {infile} -p 1 1 1 10
    out_swc_file, _ = trace_init("NeuroGPSTree", img_file, out_swc_file=out_swc_file, somamarker_file=None)
    if (os.path.exists(out_swc_file)): return
    result_name = img_file + "_NeuroGPSTree.swc"

    cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x NeuroGPSTree -f tracing_func -i {img_file} -p 1 1 1 10'

    env = os.environ.copy()
    ld_library_path = v3d_path[:-6]  # '/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin'
    env['LD_LIBRARY_PATH'] = f"{ld_library_path}:{env.get('LD_LIBRARY_PATH', '')}"
    # os.system(f"{cmd} env {env}")
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    if(os.path.exists(result_name)):
        os.rename(result_name, out_swc_file)
        # print(f"NeuroGPSTree tracing done: {out_swc_file}")
    else:
        print(f"NeuroGPSTree tracing failed: {out_swc_file}")
        # print(cmd)
        pass

# v3d_path = r"/home/user/Vaa3D_CentOS_64bit_v3.601/bin/vaa3d"
def neuTube_trace_file(img_file, out_swc_file=None, v3d_path=r"/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin/vaa3d"):
    # f'{self.vaa3d_path} -x NeuroGPSTree -f tracing_func -i {infile} -p 1 1 1 10
    out_swc_file, _ = trace_init("neuTube", img_file, out_swc_file=out_swc_file, somamarker_file=None)
    if (os.path.exists(out_swc_file)): return
    result_name = img_file + "_neutube.swc"

    cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x neuTube -f neutube_trace -i {img_file} -p 1 1'

    env = os.environ.copy()
    ld_library_path = v3d_path[:-6]  # '/home/user/Vaa3D_CentOS_64bit_v3.601/bin'
    env['LD_LIBRARY_PATH'] = f"{ld_library_path}:{env.get('LD_LIBRARY_PATH', '')}"
    # os.system(f"{cmd} env {env}")
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    if(os.path.exists(result_name)):
        os.rename(result_name, out_swc_file)
        # print(f"NeuroGPSTree tracing done: {out_swc_file}")
    else:
        print(f"neuTube tracing failed: {out_swc_file}")
        # print(cmd)
        pass

if __name__ == "__main__":
    img_file = "/data/kfchen/trace_ws/trace_consensus_test/2369.tif"
    trace_result_root = "/data/kfchen/trace_ws/trace_consensus_test"
    # out_swc_file = "/data/kfchen/trace_ws/trace_consensus_test/2369.swc"
    if (uuid.UUID(int=uuid.getnode()).hex[-12:] == "bc6ee23a04f7"):
        v3d_path = r"D:\Vaa3D-x.1.1.2_Windows_64bit\Vaa3D-x.exe"
    elif (sys.platform == "linux"):
        v3d_x_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"
        v3d_v3_path = r"/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin/vaa3d"

    file_name = os.path.basename(img_file).replace(".tif", ".swc")

    APP1_trace_file(img_file, out_swc_file=os.path.join(trace_result_root + "/APP1/" + file_name), v3d_path=v3d_x_path)
    APP2_trace_file(img_file, out_swc_file=os.path.join(trace_result_root + "/APP2/" + file_name), v3d_path=v3d_x_path)
    Advantra_trace_file(img_file, out_swc_file=os.path.join(trace_result_root + "/Advantra/" + file_name), v3d_path=v3d_x_path)
    # Meanshift_trace_file(img_file)
    # Fastmarching_spanningtree_trace_file(img_file, v3d_path=v3d_v3_path)
    CWlab_method_v1(img_file, out_swc_file=os.path.join(trace_result_root + "/Cwlab_ver1/" + file_name), v3d_path=v3d_v3_path)
    MOST_trace_file(img_file, out_swc_file=os.path.join(trace_result_root + "/MOST/" + file_name), v3d_path=v3d_v3_path)
    # TReMap_trace_file(img_file, out_swc_file=os.path.join(trace_result_root + "/TReMap/" + file_name), v3d_path=v3d_x_path) # only work on v3draw
    Mst_tracing_file(img_file, out_swc_file=os.path.join(trace_result_root + "/MST/" + file_name), v3d_path=v3d_v3_path)
