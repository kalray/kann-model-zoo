#!/usr/bin/env python3
###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

from functools import reduce
from subprocess import Popen
import collections
import glob
import importlib
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import queue

from screeninfo import get_monitors
import click
import cv2
import numpy as np
import yaml


def log(msg):
    print("[KaNN Demo] " + msg)


class SourceReader:
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
    monitors = get_monitors()
    if len(monitors) == 0:
        print("No display detected.\n")
        return None
    else:
        monitor = monitors[0]
        print("Several display detected, using the first one: H={}, W={}\n"
              .format(monitor.height, monitor.width))
        return {"size": {'h': monitor.height, 'w': monitor.width},
                "pos": {'x': 0, 'y': 0}}


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


def show_frame(window_name, frame):
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


def read_kann_output(kann_out, batching):
    # Ordered to keep the alphabetical order
    data = collections.OrderedDict()
    for name, output in kann_out.items():
        file = output['fifo']
        size = output['size']
        dtype = output['dtype']
        try:
            for batch in range(batching - 1):
                array_from_fifo(file, dtype=dtype, count=size)           # just trash the batching over 1
            data[name] = array_from_fifo(file, dtype=dtype, count=size)  # keep one output
        except:
            raise Exception("Reading of {} values in {} format from {} failed".format(size, dtype.__name__, name))
    return data

def run_demo(
        config:dict,
        generated_dir:str,
        src_reader,
        fifos_in:dict,
        fifos_out:dict,
        window_info,
        batching_:bool = False,
        display:bool = True,
        out_video:bool = False,
        out_img_path:str = None,
        verbose:bool = False
    ):
    """
    @param config   Content of <network>.yaml file.
    @param src_reader SourceReader object abstracting the source type and
                      replay mode.
    @param fifos_in The name of the fifo to use as input of kann.
    @param fifos_out The name of the fifo to use as output of kann.
    @param batching The batch size
    @param display  Enable graphical display of processed frames.
    @return         The number of frames processed.
    """
    # read the classes file, parser of classes file is done in output_preparator
    with open(config['classes_file'], 'r') as f:
        config['classes'] = f.readlines()
    log("<classes_file> at {} contains {} classes"
        .format(config['classes_file'], len(config['classes'])))

    # load the input_preparator as a python module
    sys.path.append(generated_dir)
    if len(config['input_preparators']) > 1:
        raise Exception("Provided network requires {} input preparators. "
                        "Only network with 1 input preparator are supported.".format(
                        len(config['input_preparators'])))
    prepare = __import__(os.path.relpath(config['input_preparators'][0])[:-3])
    output_preparator = importlib.import_module(
        os.path.relpath(config['output_preparator']).replace('/', '.') + '.output_preparator')

    if isinstance(config['forced_batch_size'], int):
        batching = config['forced_batch_size']
    else:
        batching = batching_

    # Open the fifo to interact with kann
    # Ordered to keep the alphabetical order
    kann_in = collections.OrderedDict()
    kann_out = collections.OrderedDict()
    buffers = sorted(config['input_nodes_name'] + config['output_nodes_name'])
    for b in buffers:
        if b in config['input_nodes_name']:
            log("Opening input fifo for CNN's input : '{}'".format(b))
            kann_in[b] = {'fifo': os.fdopen(os.open(fifos_in[b],  os.O_WRONLY), 'wb', 0)}
            kann_in[b]['dtype'] = getattr(np, config['input_nodes_dtype'][config['input_nodes_name'].index(b)])
    for b in buffers:
        if b in config['output_nodes_name']:
            log("Opening output fifo for CNN's output : '{}'".format(b))
            kann_out[b] = {'fifo': os.fdopen(os.open(fifos_out[b], os.O_RDONLY), 'rb')}
    for b, shape, dtype in zip(config['output_nodes_name'], config['output_nodes_shape'], config['output_nodes_dtype']):
        kann_out[b]['size'] = reduce(lambda x, y: x*y, shape)
        kann_out[b]['dtype'] = getattr(np, dtype)

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

    nframes = int(src_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t = [0] * 8
    frames_counter = 0
    frame = prev_frame = None
    out = None
    while True:
        t[0] = time.perf_counter() # CATCH FRAME ###############################

        prev_frame = frame
        frame = src_reader.get_frame()
        if frame is None:
            break
        frames_counter += 1

        t[1] = time.perf_counter() # PRE-PROCESS FRAME #########################

        # For now, we are supporting multiple input, but only one source per
        # instance, thus the input preparator prepare all the inputs base on one
        # source.
        # TODO : support multiple sources and multiple inputs
        prepared = prepare.prepare_img(frame)
        if not isinstance(prepared, (tuple, list)):
            prepared = [prepared]
        assert len(prepared) == len(kann_in)

        t[2] = time.perf_counter() # SEND TO KANN RUNTIME ######################

        # Risky since we can't ensure the order that the input preparator output
        # the data in the same order that alphabetical order.
        # TODO : Manage input and sources
        for p, i in zip(prepared, kann_in.values()):
            assert p.dtype == i['dtype'], \
                "Pre processed image is in {} format " \
                "but {} is expected".format(p.dtype, i['dtype'].__name__)
            p.tofile(i['fifo'], '')

        t[3] = time.perf_counter() # READ PROCESSED FRAME ##################
        out = read_kann_output(kann_out, batching)

        t[4] = time.perf_counter() # POST-PROCESS FRAME ####################
        frame = output_preparator.post_process(
            config, frame, out, device='mppa', dbg=verbose)

        t[5] = time.perf_counter()  # ANNOTATE FRAME #######################
        annotate_frame(frame, t[4] - t[3], config['name'])  # FPS: KaNN

        t[6] = time.perf_counter()  # DISPLAY FRAME ########################
        if display:
            if not show_frame(window_name, frame):
                break

        t[7] = time.perf_counter() # END ###################################
        log("frame:{}/{}\tread: {:0.2f}ms\tpre: {:0.2f}ms\tsend: {:0.2f}ms\t"
            "kann: {:0.2f}ms\tpost: {:0.2f}ms\tdraw: {:0.2f}ms\t"
            "show: {:0.2f}ms\ttotal: {:0.2f}ms ({:0.1f}fps, kann:{:0.1f}fps)".format(
            frames_counter + 1, nframes,
            1000*(t[1]-t[0]),  # read (ms)
            1000*(t[2]-t[1]),  # preprocessing (ms)
            1000*(t[3]-t[2]),  # send data to pipe (ms)
            1000*(t[4]-t[3]),  # kann + read data from pipe (ms)
            1000*(t[5]-t[4]),  # post processing (ms)
            1000*(t[6]-t[5]),  # annotate frame (ms)
            1000*(t[7]-t[6]),  # show frame + post processed predictions (ms)
            1000*(t[7]-t[0]),  # total (ms)
            1. / (t[7]-t[0]),  # total (fps)
            1. / (t[4]-t[3]))  # kann + read data from pipe (fps)
        )

        if out_video is not None:
            out_video.write(frame)

    if display:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1) # pump all events, avoid bug in opencv where windows are not properly closed
    if out_img_path:
        cv2.imwrite(out_img_path, prev_frame)
        log(f"Last frame has been saved to: {out_img_path}")
    # close the input, to initiate the terminaison sequence in kann
    for i in kann_in.values():
        i['fifo'].close()
    for o in kann_out.values():
        o['fifo'].close()
    return frames_counter


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument(
    'generated-dir',
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    required=True)
@click.argument(
    'source',
    type=click.STRING,
    required=True)
@click.option(
    '--binaries-dir',
    type=click.Path(exists=True, file_okay=False),
    help="Path to the directory containing the compiled binaries. It should "
         "contain the files kann_opencl_cnn. If you have compiled your CNNs "
         "with multi-generic cnn, binaries_dir is "
         "examples/app/opencl_generic_cnn/output/bin.")
@click.option(
    '--kernel-binaries-dir',
    type=click.Path(exists=True, file_okay=False),
    help="Path to the directory containing the compiled binaries. It should "
         "contain the OpenCL kernel binaries, mppa_kann_opencl.cl.pocl. If you "
         "have compiled your CNNs with openc example genericcnn, binaries_dir "
         "is examples/app/opencl_generic_cnn/output/bin.")
@click.option(
    '--local-install-dir',
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="A custom local_install_dir can be specified to use a custom KaNN "
         "runtime installation.")
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
    '--save-video',
    is_flag=True,
    help="Save input video with output predictions as video file.")
@click.option(
    '--save-img',
    is_flag=True,
    help="Save last frame with output predictions as video file.")
@click.option(
    '--batching',
    type=int,
    default=1,
    help="Set the batch size, default batching is 1.")
def main(generated_dir,
         source,
         binaries_dir,
         kernel_binaries_dir,
         local_install_dir,
         no_display,
         no_replay,
         save_video,
         save_img,
         batching,
         verbose):
    """ Kalray Neural Network demonstrator.

    GENERATED_DIR is a generated network folder.
    SOURCE is an input video. It can be either:
    \t- A webcam ID, typically 0 on a machine with a single webcam.
    \t- A video file in a format supported by OpenCV.
    \t- An image sequence (eg. img_%02d.jpg, which will read samples like
    img_00.jpg, img_01.jpg, img_02.jpg, ...).
    """

    # find <network>.yaml file in generated_dir
    config_files = glob.glob(os.path.join(generated_dir, "*.yaml"))
    if not config_files:
        log("{}/<network>.yaml no such file".format(generated_dir))
        sys.exit(1)
    elif len(config_files) > 1:
        log("Found multiple candidates for {}/<network>.yaml"
            .format(generated_dir))
        sys.exit(1)
    config_file = config_files[0]

    # find serialized_params_<CNN_name>.kann file in generated_dir
    params_files = glob.glob(os.path.join(
        generated_dir, "*.kann"))
    if not params_files:
        log("{}/<CNN_name>.kann no such file"
            .format(generated_dir))
        sys.exit(1)
    elif len(params_files) > 1:
        log("Found multiple candidates for {}/<CNN_name>.kann"
            .format(generated_dir))
        sys.exit(1)
    serialized_params_file = params_files[0]

    if binaries_dir is None:
        binaries_dir = "$KALRAY_TOOLCHAIN_DIR/bin"
        binaries_dir = os.path.expandvars(binaries_dir)
    if not os.path.isfile(os.path.join(binaries_dir, 'kann_opencl_cnn')):
        log("kann_opencl_cnn must be present in <binaries_dir> {}"
            .format(binaries_dir))
        sys.exit(1)

    if kernel_binaries_dir is None:
        kernel_binaries_dir = "$KALRAY_TOOLCHAIN_DIR/kvx-cos/lib/KAF/services"
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
        src_reader = SourceReader(source, not no_replay)
    except Exception as e:
        log("ERROR: {}".format(e))
        sys.exit(1)

    if save_video:
        out_video_path = './{}.avi'.format(os.path.basename(source).split('.')[0])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(
            out_video_path, fourcc, 25., (src_reader.width, src_reader.height))
    else:
        out_video = None

    if save_img:
        out_img_path = './{}.jpg'.format(os.path.basename(source).split('.')[0])
    else:
        out_img_path = None

    custom_env = ''
    if local_install_dir is not None:
        custom_env = os.path.join(local_install_dir, 'lib')

    # load config file
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    extra_data = config['extra_data']
    config['classes_file'] = os.path.join(generated_dir, extra_data['classes'])
    config['input_preparators'] = extra_data['input_preparators']
    config['output_preparator'] = extra_data['output_preparator']
    if not "input_nodes_dtype" in config:
        config['input_nodes_dtype'] = ["float32"] * len(config['input_nodes_name'])
    if not "output_nodes_dtype" in config:
        config['output_nodes_dtype'] = ["float32"] * len(config['output_nodes_name'])
    assert len(config['input_nodes_dtype']) == len(config['input_nodes_name'])
    assert len(config['output_nodes_dtype']) == len(config['output_nodes_name'])

    # create the kann_fifo_{in,out} in a new temporary directory
    # this allows to remove cleanly the fifos once the program terminate
    # (and it is easier than creating every fifos and conditionnaly remove the
    # ones that failed to open)
    fifos_dir = tempfile.mkdtemp()
    kann_proc = None
    try:
        log("Temporary directory for the fifos is {}".format(fifos_dir))

        if os.path.exists(fifos_dir):
            shutil.rmtree(fifos_dir)

        fifos_in = {}
        for input_ in config['input_nodes_name']:
            input_path = fifos_dir + "/{}".format(input_)
            dir = os.path.dirname(input_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            os.mkfifo(input_path)
            fifos_in[input_] = input_path

        fifos_out = {}
        for output in config['output_nodes_name']:
            output_path = fifos_dir + "/{}".format(output)
            dir = os.path.dirname(output_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            os.mkfifo(output_path)
            fifos_out[output] = output_path

        # start the KANN program
        kann_args = [
            os.path.join(binaries_dir, "kann_opencl_cnn"),
            serialized_params_file,
            fifos_dir,
        ]
        log("Spawning kann with command: " + ' '.join(v for v in kann_args))

        os.environ["LD_LIBRARY_PATH"] = custom_env + ":" + os.environ["LD_LIBRARY_PATH"]
        if not os.environ.get("POCL_CACHE_DIR"):
            os.environ["POCL_CACHE_DIR"] = os.path.abspath(os.path.expandvars("$HOME/.pocl_cache_dir"))

        # Do not use opencl to offload opencv (conflicting with kann on mppa)
        # ref T12057
        os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
        os.environ["OPENCV_OPENCL_RUNTIME"] = "null"
        kann_proc = Popen(kann_args, bufsize=-1,
            env=dict(os.environ, KANN_POCL_FILE=os.path.join(kernel_binaries_dir, "mppa_kann_opencl.cl.pocl")))
        # Manage window position and size
        window_info = None if no_display else getTiledWindowsInfo()
        assert (no_display or window_info is not None)

        # run demo
        run_demo(
            config,
            generated_dir,
            src_reader,
            fifos_in,
            fifos_out,
            window_info,
            batching,
            not no_display,
            out_video,
            out_img_path,
            verbose)
    except Exception as e:
        log("ERROR:\n" + traceback.format_exc())
    finally:
        # make sure we kill kann no matter what happens
        # (most of the time: videofile unexpectedly closed, or
        # kann took more than 2s to terminate after we closed kann_fifo_in)
        if save_video:
            out_video.release()
            log("Output has been save to {}".format(out_video_path))
        if isinstance(kann_proc, Popen):
            try:
                # give time to kann to exit properly
                kann_proc.wait(timeout=5)
            except:
                log("Killing kann process")
                kann_proc.terminate()
        log("Removing temporary directory {}".format(fifos_dir))
        os.system("rm -rf {}".format(fifos_dir))


if __name__ == '__main__':
    main()
