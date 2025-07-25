#!/usr/bin/env python3
###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import cv2
import sys
import time
import yaml
import json
import glob
import click
import queue
import shutil
import signal
import tempfile
import threading
import importlib
import collections
import numpy as np

from functools import reduce
from subprocess import Popen


def log(msg):
    print("[KaNN Demo] " + msg)


class SourceReader(object):
    def __init__(self, source, replay):
        self.source = source
        self.replay = replay
        self.is_camera = isinstance(self.source, int)

        self.cap = cv2.VideoCapture(self.source)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        log("Video backend: {}".format(self.cap.getBackendName()))
        if not self.cap.isOpened():
            raise Exception("Cannot open video source {}".format(self.source))

        self._frame_queue = queue.Queue(1)
        if self.is_camera:
            self._thread = threading.Thread(target=self._decode_camera)
        else:
            self._thread = threading.Thread(target=self._decode_file)

    def start_decode(self):
        self._thread.start()

    def get_frame(self):
        while self._thread.is_alive():
            try:
                return self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                pass
        return None

    def _decode_camera(self):
        while self.cap.isOpened() and threading.main_thread().is_alive():
            ret, frame = self.cap.read()
            if not ret:
                frame = None
                log("Camera stream ended (it could have been disconnected)")

            # drop any previous image before publishing a new one
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put(frame)

    def _decode_file(self):
        while self.cap.isOpened() and threading.main_thread().is_alive():
            ret, frame = self.cap.read()
            if not ret:
                frame = None
                self.cap.release()
                if self.replay:
                    log("Looping over video file, use --no-replay to play "
                        "video only once.")
                    self.cap = cv2.VideoCapture(self.source)
                    ret, frame = self.cap.read()
                    if not ret:
                        raise Exception("Cannot loop over {}"
                            .format(self.source))

            # wait for previous image to be consumed and loop over a timeout to
            # eventually exit with main thread
            while threading.main_thread().is_alive():
                try:
                    self._frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    continue # previous image is still there, keep waiting...
                break


def getTiledWindowsInfo():
    from screeninfo import get_monitors
    try:
        monitor = get_monitors()[0]
        log("Several display detected, using the first one: H={}, W={}\n"
            .format(monitor.height, monitor.width))
        return {"size": {'h': monitor.height, 'w': monitor.width},
                "pos": {'x': 0, 'y': 0}}
    except:
        log("[WARNING] ** Screen or Display has not been found **\n")
        return None


def draw_text(frame, lines, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = frame.shape[0] / 640
    thick = 1
    white = (255, 255, 255)
    black = (0, 0, 0)

    textsize, baseline = cv2.getTextSize(lines[0], font, scale, thick)
    x1, y1 = pos[0], pos[1]
    x2, y2 = pos[0] + textsize[0] + baseline, pos[1] - (textsize[1] + baseline) * len(lines)
    cv2.rectangle(frame, (x1, y1), (x2, y2), white, -1)
    for i, line in enumerate(lines):
        origin = (pos[0], (pos[1] - (textsize[1] + baseline) * i))
        cv2.putText(frame, line, origin, font, scale, black, thick, cv2.LINE_AA)


def annotate_frame(frame, delta_t, title):
    framerate = 1.0 / delta_t
    lines = ["Algorithm: {:15s}".format(title)]
    lines += ["Speed: {:.1f} fps".format(framerate)]
    origin = (10, frame.shape[0] - 10)
    draw_text(frame, lines, origin)


def show_frame(config):
    for nn in config:
        window_name = config[nn]['window_name']
        frame = config[nn]['frame']
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False  # window closed
        cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27:  # wait for 1ms
        return False  # escape key
    return True


def array_from_fifo(fd, dtype, count):
    nb_bytes = np.dtype((dtype, count)).itemsize
    buf = b''
    while nb_bytes > 0:
        tmp_buf = fd.read(nb_bytes)
        nb_read = len(tmp_buf)
        if nb_read == 0:
            raise Exception("Read failed, EOF or pipe closed")
        nb_bytes -= nb_read
        buf += tmp_buf
    return np.frombuffer(buf, dtype)


def read_kann_output(kann_out):
    # Ordered to keep the alphabetical order
    data = collections.OrderedDict()
    for name, output in kann_out.items():
        file = output['fifo']
        size = output['size']
        dtype = output['dtype']
        try:
            data[name] = array_from_fifo(file, dtype=dtype, count=size)  # keep one output
        except:
            raise Exception("Reading of {} values in {} format from {} failed".format(size, dtype.__name__, name))
    return data


def run_demo(
        networks : dict,
        src_reader,
        window_info,
        display : bool = True,
        nb_frames : int = -1,
        out_img_path : bool = False,
        verbose : bool = False
    ):

    for nn in networks:
        config = networks[nn]

        # read the classes file, parser of classes file is done in output_preparator
        with open(config['classes_file'], 'r') as f:
            config['classes'] = f.readlines()
        log("({}) <classes_file> at {} contains {} classes"
            .format(config['name'], config['classes_file'], len(config['classes'])))

        # load the input_preparator as a python module
        generated_dir = os.path.dirname(config['serialized'])
        sys.path.append(os.path.abspath(os.path.dirname(generated_dir)))

        if len(config['input_preparators']) > 1:
            raise Exception("Provided network requires {} input preparators. "
                            "Only network with 1 input preparator are supported.".format(
                            len(config['input_preparators'])))

        pre_proc =  os.path.relpath(config['input_preparators'][0]).replace('/', '.')[:-3]
        pre_proc_module = pre_proc.split(".")[-1]
        prepare = importlib.import_module(pre_proc, pre_proc_module)

        post_proc =  os.path.relpath(config['output_preparator']).replace('/', '.') + '.output_preparator'
        post_proc_module = pre_proc.split(".")[-1]
        output_preparator = importlib.import_module(post_proc, post_proc_module)

        config['fn_prepare'] = prepare.prepare_img
        config['fn_post_process'] = output_preparator.post_process

        # Open the fifo to interact with kann
        # Ordered to keep the alphabetical order
        fifos_in = config['fifos_in']
        fifos_out = config['fifos_out']
        kann_in = collections.OrderedDict()
        kann_out = collections.OrderedDict()
        buffers = sorted(config['input_nodes_name'] + config['output_nodes_name'])
        for b in buffers:
            if b in config['input_nodes_name']:
                log("({}) Opening input fifo for CNN's input : '{}'".format(config['name'], b))
                kann_in[b] = {'fifo': os.fdopen(os.open(fifos_in[b],  os.O_WRONLY), 'wb', 0)}
                kann_in[b]['dtype'] = getattr(np, config['input_nodes_dtype'][config['input_nodes_name'].index(b)])
        for b in buffers:
            if b in config['output_nodes_name']:
                log("({}) Opening output fifo for CNN's output : '{}'".format(config['name'], b))
                kann_out[b] = {'fifo': os.fdopen(os.open(fifos_out[b], os.O_RDONLY), 'rb')}
        for b, shape, dtype in zip(config['output_nodes_name'], config['output_nodes_shape'], config['output_nodes_dtype']):
            kann_out[b]['size'] = reduce(lambda x, y: x * y, shape)
            kann_out[b]['dtype'] = getattr(np, dtype)

        config['kann_in'] = kann_in
        config['kann_out'] = kann_out

        window_name = config['name']
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(window_name, window_info['pos']['x'], window_info['pos']['y'])
            win_size = 640
            ratio = src_reader.width / src_reader.height
            if src_reader.width >= src_reader.height:
                cv2.resizeWindow(window_name, win_size, int(win_size / ratio))
                log("Source frame is W{}xH{}, OpenCV window is resized to {}x{}".format(
                    src_reader.width, src_reader.height, win_size, int(win_size / ratio)))
            else:
                cv2.resizeWindow(window_name, win_size, int(win_size * ratio))
                log("Source frame is W{}xH{}, OpenCV window is resized to {}x{}".format(
                    src_reader.width, src_reader.height, int(win_size * ratio), win_size))
        config['window_name'] = window_name

    nframes = int(src_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t = [0] * 8
    frames_counter = 0
    frame = None

    src_reader.start_decode()

    while True:  # infinite loop

        # CATCH FRAME ############################
        t[0] = time.perf_counter()
        prev_frame = frame
        frame = src_reader.get_frame()
        if frame is None:
            break
        frames_counter += 1

        # PRE-PROCESS FRAME ######################
        t[1] = time.perf_counter()
        for nn in networks:
            networks[nn]['prepared'] = networks[nn]["fn_prepare"](frame)
            if not isinstance(networks[nn]['prepared'], (tuple, list)):
                networks[nn]['prepared'] = [networks[nn]['prepared']]
            assert len(networks[nn]['prepared']) == len(networks[nn]['kann_in'])

        # SEND TO KANN RUNTIME ###################
        t[2] = time.perf_counter()
        for nn in networks:
            for p, i in zip(networks[nn]['prepared'], networks[nn]['kann_in'].values()):
                assert p.dtype == i['dtype'], \
                    "Pre processed image is in {} format " \
                    "but {} is expected".format(p.dtype, i['dtype'].__name__)
                try:
                    p.tofile(i['fifo'], '')
                except:
                    return

        # READ PROCESSED FRAME ####################
        t[3] = time.perf_counter()
        for nn in networks:
            networks[nn]['out'] = read_kann_output(networks[nn]['kann_out'])

        # POST-PROCESS FRAME ######################
        t[4] = time.perf_counter()
        for nn in networks:
            networks[nn]['frame'] = networks[nn]["fn_post_process"](
                networks[nn], frame.copy(), networks[nn]['out'], device='mppa', dbg=verbose)

        # ANNOTATE FRAME ##########################
        t[5] = time.perf_counter()
        for nn in networks:
            annotate_frame(networks[nn]['frame'], t[4] - t[3], networks[nn]['name'])

        # DISPLAY FRAME ###########################
        t[6] = time.perf_counter()
        if cv2.waitKey(1) == 27:  # wait for 1ms
            log("Escape key pressed, exiting...")
            break
        if display:
            status = True
            for nn in networks:
                status = show_frame(networks)
            if not status:
                break

        # PRINT TIMINGS ###########################
        t[7] = time.perf_counter()
        log("frame:{}/{}\tread: {:0.2f}ms\tpre: {:0.2f}ms\tsend: {:0.2f}ms\t"
            "kann: {:0.2f}ms\tpost: {:0.2f}ms\tdraw: {:0.2f}ms\t"
            "show: {:0.2f}ms\ttotal: {:0.2f}ms ({:0.1f}fps, kann:{:0.1f}fps)".format(
            frames_counter, nframes,
            1000*(t[1]-t[0]),  # read (ms)
            1000*(t[2]-t[1]),  # preprocessing (ms)
            1000*(t[3]-t[2]),  # send data to pipe (ms)
            1000*(t[4]-t[3]),  # kann + read data from pipe (ms)
            1000*(t[5]-t[4]),  # post processing (ms)
            1000*(t[6]-t[5]),  # annotate frame (ms)
            1000*(t[7]-t[6]),  # show frame (ms)
            1000*(t[7]-t[0]),  # total (ms)
            1. / (t[7]-t[0]),  # total (fps)
            1. / (t[4]-t[3]))  # kann + read data from pipe (fps)
        )

        # END #####################################
        # looping or not
        if nb_frames < 0:
            continue
        elif frames_counter >= nb_frames:
            break
        # end of while loop

    if display:
        log("Closing all OpenCV windows")
        cv2.destroyAllWindows()

    if out_img_path:
        for nn in networks:
            if networks[nn]['frame'] is None:
                networks[nn]['frame'] = prev_frame
            cv2.imwrite(networks[nn]['out_img_path'], networks[nn]['frame'])
            log(f"Last frame has been saved to: {networks[nn]['out_img_path']}")

    # Close the FIFOs, to initiate the terminaison sequence in kann
    for nn in networks:
        kann_in = networks[nn]['kann_in']
        kann_out = networks[nn]['kann_out']
        for i in kann_in.values():
            i['fifo'].close()
        for o in kann_out.values():
            o['fifo'].close()

    return frames_counter


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument(
    'generated-dir',
    type=click.Path(exists=True, file_okay=False),
    nargs=-1,
    required=True)
@click.argument(
    'source',
    type=click.Path(exists=True, file_okay=True),
    required=True)
@click.option(
    '--bin-file',
    type=click.Path(exists=True, file_okay=True),
    help="Path to compiled binary file.")
@click.option(
    '--kernel-binaries-dir',
    type=click.Path(exists=True, file_okay=False),
    help="Path to the directory containing the compiled binaries. It should "
         "contain the OpenCL kernel binaries, mppa_kann_opencl.cl.pocl. If you "
         "have compiled your CNNs with openc example genericcnn, binaries_dir "
         "is examples/app/opencl_generic_cnn/output/bin.")
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help="Display detection and time spent into post-process tasks")
@click.option(
    '--no-display',
    is_flag=True,
    help="Disable graphical display.")
@click.option(
    '--no-replay',
    is_flag=True,
    help="Disable video loop if source is a video file.")
@click.option(
    '--save-img',
    is_flag=True,
    help="Save last frame with output predictions as video file.")
@click.option(
    '--nb-frames', '-n',
    type=int,
    default=-1,
    help="Run inference on N frames only")
def main(generated_dir,
         source,
         bin_file,
         kernel_binaries_dir,
         no_display,
         no_replay,
         save_img,
         nb_frames,
         verbose):
    """ Kalray Neural Network demonstrator for Multi-CNN instance.

    GENERATED_DIR is a generated network folder.
    SOURCE is a stream. It can be either:
    \t- A webcam ID, typically 0 on a machine with a single webcam.
    \t- A video file in a format supported by OpenCV.
    \t- An image sequence (eg. img_%02d.jpg, which will read samples like
    img_00.jpg, img_01.jpg, img_02.jpg, ...).
    """

    # find <network>.yaml file in generated_dir
    networks = {gen_dir: glob.glob(os.path.join(gen_dir, "*.yaml"))[0] for gen_dir in generated_dir}
    if len(networks.values()) == 0:
        log("{}/<network>.yaml no such file".format(generated_dir))
        sys.exit(1)

    for nn in networks:
        config_file = networks[nn]
        # find inputs_outputs_info.json file in generated_dir
        nn_dir = os.path.dirname(networks[nn])
        json_file = glob.glob(os.path.join(nn_dir, "inputs_outputs_info.json"))[0]
        if not json_file:
            log("inputs_outputs_info.json file not found inside the generated dir {}".format(nn_dir))
            sys.exit(1)
        # load config file
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        extra_data = config['extra_data']
        config['classes_file'] = os.path.join(nn_dir, extra_data['classes'])
        config['input_preparators'] = [os.path.join(nn, a) for a in extra_data['input_preparators']]
        config['output_preparator'] = os.path.join(nn, extra_data['output_preparator'])
        with open(json_file, 'r') as j:
            io = json.load(j)
            for key in io:
                if not config[key]:
                    config[key] = io[key]
        assert len(config['input_nodes_dtype']) == len(config['input_nodes_name'])
        assert len(config['output_nodes_dtype']) == len(config['output_nodes_name'])

        # find serialized_params_<CNN_name>.kann file in generated_dir
        params_files = glob.glob(os.path.join(nn_dir, "*.kann"))
        if not params_files:
            log("{}/<CNN_name>.kann no such file".format(nn_dir))
            sys.exit(1)
        serialized_params_file = params_files[0]

        networks[nn] = config
        networks[nn]['serialized'] = serialized_params_file 

    if bin_file is None:
        binaries_dir = "$KALRAY_TOOLCHAIN_DIR/bin"
        binaries_dir = os.path.expandvars(binaries_dir)
        bin_file = os.path.join(binaries_dir, 'kann_opencl_cnn')
        if not os.path.isfile(bin_file):
            log("kann_opencl_cnn must be present in <binaries_dir> {}"
                .format(binaries_dir))
            sys.exit(1)

    if kernel_binaries_dir is None:
        kernel_binaries_dir = "$KALRAY_TOOLCHAIN_DIR/kvx-cos/lib/kv3-2/KAF/services"
        kernel_binaries_dir = os.path.expandvars(kernel_binaries_dir)
    if not os.path.isfile(os.path.join(kernel_binaries_dir,
            'mppa_kann_opencl.cl.pocl')):
        log("mppa_kann_opencl.cl.pocl must be present in <kernel_binaries_dir> "
            "{}".format(kernel_binaries_dir))
        sys.exit(1)

    # convert source argument to int if it is a webcam index
    if source.isdigit():
        source = int(source)
    try:
        src_reader = SourceReader(source, not no_replay or nb_frames > 0)
    except Exception as e:
        log("ERROR: {}".format(e))
        sys.exit(1)

    # Define the output image path if save_img is True
    # -- 
    # for each networks 
    # Create the kann_fifo_{in,out} in a new temporary directory
    # this allows to remove cleanly the fifos once the program terminate
    # (and it is easier than creating every fifos and conditionnaly remove the
    # ones that failed to open)
    kann_proc = None
    args = []
    for nn in networks:
        if save_img:
            file_name = os.path.basename(source).split('.')[0]
            networks[nn]['out_img_path'] = f"{nn}_{file_name}.jpg"
        else:
            networks[nn]['out_img_path'] = None

        fifos_dir = tempfile.mkdtemp()
        networks[nn]['fifos_dir'] = fifos_dir
        log("[Python {}] Temporary directory for the fifos is {}".format(nn, fifos_dir))
        if os.path.exists(fifos_dir):
            shutil.rmtree(fifos_dir)

        fifos_in = {}
        for input_ in networks[nn]['input_nodes_name']:
            input_path = fifos_dir + "/{}".format(input_)
            dir = os.path.dirname(input_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            os.mkfifo(input_path)
            fifos_in[input_] = input_path
        networks[nn]['fifos_in'] = fifos_in

        fifos_out = {}
        for output in networks[nn]['output_nodes_name']:
            output_path = fifos_dir + "/{}".format(output)
            dir = os.path.dirname(output_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            os.mkfifo(output_path)
            fifos_out[output] = output_path
        networks[nn]['fifos_out'] = fifos_out
        # define args for binary
        args += [networks[nn]['serialized'], networks[nn]['fifos_dir']]

    kann_args = [os.path.abspath(bin_file)] + args
    log("Spawning kann with command: " + ' '.join(v for v in kann_args))

    os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    if not os.environ.get("POCL_CACHE_DIR"):
        os.environ["POCL_CACHE_DIR"] = os.path.abspath(os.path.expandvars("$HOME/.pocl_cache_dir"))

    # Do not use opencl to offload opencv (conflicting with kann on mppa)
    # ref T12057
    os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
    os.environ["OPENCV_OPENCL_RUNTIME"] = "null"
    kann_proc = Popen(kann_args, bufsize=-1,
        env=dict(os.environ, KANN_POCL_FILE=os.path.join(kernel_binaries_dir, "mppa_kann_opencl.cl.pocl")))

    # Manage window position and size
    window_info = getTiledWindowsInfo()
    if window_info is None:
        no_display = True

    def handle_sigint(signum, frame):
        log("\nSIGINT (Ctrl+C) received. Exiting gracefully...")
        kann_proc.wait(timeout=5)
        kann_proc.terminate()
        log("Removing temporary directory {}".format(fifos_dir))
        for nn in networks:
            os.system("rm -rf {}".format(networks[nn]['fifos_dir']))
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        # run demo
        run_demo(
            networks,
            src_reader,
            window_info,
            not no_display,
            nb_frames,
            save_img,
            verbose)

    finally:
        # make sure we kill kann no matter what happens
        # most of the time: videofile unexpectedly closed, or
        # kann took more than 2s to terminate after we closed kann_fifo_in
        if isinstance(kann_proc, Popen):
            try:
                # give time to kann to exit properly
                kann_proc.wait(timeout=5)
            except:
                log("Killing kann process")
                kann_proc.terminate()
        log("Removing temporary directory {}".format(fifos_dir))
        for nn in networks:
            os.system("rm -rf {}".format(networks[nn]['fifos_dir']))


if __name__ == '__main__':
    main()
