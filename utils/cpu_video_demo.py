#!/usr/bin/env python3
###
# Copyright (C) 2025 Kalray SA. All rights reserved.
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
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import queue
import numpy

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


def show_frame(window_name, frame):
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return False # window closed
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27: # wait for 1ms
        return False # escape key
    return True


def run_demo(
        config : dict,
        network_dir : str,
        src_reader,
        window_info,
        display : bool = True,
        out_video=None,
        out_img_path : str = None,
        verbose=True
    ):
    """
    @param config         Content of <network>.yaml file.
    @param network_dir    Generated dir from KaNN
    @param src_reader     SourceReader object abstracting the source type and
                          replay mode.
    @param window_info    Information about the available screens provided by screeninfo
    @param display        Enable graphical display of processed frames.
    @param out_video      Write into an output file the video stream
    @param out_img_path   Write into an output file the last frame processed

    @return               The number of frames processed.
    """

    global sess

    # read the classes file, parser of classes file is done in output_preparator
    with open(config['classes_file'], 'r') as f:
        config['classes'] = f.readlines()
    log("<classes_file> at {} contains {} classes"
        .format(config['classes_file'], len(config['classes'])))

    # load the input_preparator as a python module
    sys.path.append(network_dir)
    prepare = __import__(config['input_preparator'][:-3])
    output_preparator_lib_name = re.sub('[^A-Za-z0-9_.]+', '', config['output_preparator']) + '.output_preparator'
    output_preparator = importlib.import_module(output_preparator_lib_name)

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
    t = [0] * 7
    frames_counter = 0
    frame = None

    # Define inputs and outputs neural network name
    inputs_name = sess.get_inputs()[0].name
    outputs_name = [o.name for o in sess.get_outputs()]

    # Infinite Loop on input stream
    while True:
        # CATCH FRAME ############################
        t[0] = time.perf_counter()
        prev_frame = frame
        frame = src_reader.get_frame()
        if frame is None:
            break
        frames_counter += 1

        # PRE-PROCESS FRAME ######################
        t[1] = time.perf_counter()
        prepared = prepare.prepare_img(frame)
        while len(prepared.shape) < 4:
            prepared = numpy.expand_dims(prepared, axis=0)
        prepared = numpy.transpose(prepared, (0, 3, 1, 2))
        ort_inputs = {inputs_name: prepared}  # work with only 1 input

        # SEND TO ONNX RUNTIME ###################
        t[2] = time.perf_counter()
        outs = sess.run(None, ort_inputs)
        out = {k: o for o, k in zip(outs, outputs_name)}

        # POST-PROCESS FRAME #####################
        t[3] = time.perf_counter()
        frame = output_preparator.post_process(
            config, frame, out, device='cpu', dbg=verbose)

        # ANNOTATE FRAME #########################
        t[4] = time.perf_counter()
        annotate_frame(frame, t[3] - t[2], config['name'])

        # DISPLAY FRAME ##########################
        t[5] = time.perf_counter()
        if display and not show_frame(window_name, frame):
            break

        # PRINT TIMINGS ##########################
        t[6] = time.perf_counter()
        log("frame:{}/{}\tread: {:0.2f}ms\tpre: {:0.2f}ms\t"
            "onnx: {:0.2f}ms\tpost: {:0.2f}ms\tdraw: {:0.2f}ms\t"
            "show: {:0.2f}ms\ttotal: {:0.2f}ms ({:0.1f}fps)".format(
            frames_counter, nframes,
            1000*(t[1]-t[0]),  # read (ms)
            1000*(t[2]-t[1]),  # preprocessing (ms)
            1000*(t[3]-t[2]),  # compute data w/ onnx session (ms)
            1000*(t[4]-t[3]),  # post processing (ms)
            1000*(t[5]-t[4]),  # annotate frame (ms)
            1000*(t[6]-t[5]),  # show frame (ms)
            1000*(t[6]-t[0]),  # total (ms)
            1.0/(t[6]-t[0]))   # total (fps)
        )
        # END ####################################
        if out_video is not None:
            out_video.write(frame)
        # end of while loop
    if display:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # pump all events, avoid bug in opencv where windows are not properly closed
    if out_img_path:
        cv2.imwrite(out_img_path, prev_frame)
        log(f"Last frame has been saved to: {out_img_path}")
    return frames_counter


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument(
    'network_config',
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
    required=True)
@click.argument(
    'source',
    type=click.STRING,
    required=True)
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
def main(
        network_config,
        source,
        verbose,
        no_display,
        no_replay,
        save_video,
        save_img):

    """ ONNX demonstrator.

    NETWORK_CONFIG is a network configuration file for KaNN generation.
    SOURCE is an input video. It can be either:
    \t- A webcam ID, typically 0 on a machine with a single webcam.
    \t- A video file in a format supported by OpenCV.
    \t- An image sequence (eg. img_%02d.jpg, which will read samples like
    img_00.jpg, img_01.jpg, img_02.jpg, ...).
    """

    global sess

    # find <network>.yaml file in generated_dir
    if not os.path.exists(network_config):
        log("{}/<network>.yaml no such file".format(network_config))
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

    # load config file
    network_dir = os.path.dirname(network_config)
    with open(network_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    extra_data = config['extra_data']
    config['classes_file'] = os.path.join(network_dir, extra_data['classes'])
    config['input_preparator'] = extra_data['input_preparators'][0]
    config['output_preparator'] = extra_data['output_preparator']
    if not "input_nodes_dtype" in config:
        config['input_nodes_dtype'] = ["float32"] * len(config['input_nodes_name'])
    if not "output_nodes_dtype" in config:
        config['output_nodes_dtype'] = ["float32"] * len(config['output_nodes_name'])
    assert len(config['input_nodes_dtype']) == len(config['input_nodes_name'])
    assert len(config['output_nodes_dtype']) == len(config['output_nodes_name'])

    try:
        # start the ONNX model
        import onnx
        import onnxruntime
        onnx_path = os.path.join(os.path.dirname(network_config), config['onnx_model'])
        if not os.path.isfile(onnx_path):
            onnx_path = os.path.join(os.path.dirname(network_config), "optimized-model.onnx")
        if not os.path.isfile(onnx_path):
            raise RuntimeError(f"{onnx_path} does not exists, please ensure that model file path exists")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        sess = onnxruntime.InferenceSession(onnx_path)
        os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
        os.environ["OPENCV_OPENCL_RUNTIME"] = "null"

        # Manage window position and size
        window_info = getTiledWindowsInfo()
        if window_info is None:
            no_display = True
        # run demo
        nbr_frames = run_demo(
            config,
            network_dir,
            src_reader,
            window_info,
            not no_display,
            out_video,
            out_img_path,
            verbose)

    except Exception as e:
        log("ERROR:\n" + traceback.format_exc())
    finally:
        # make video file unexpectedly closed
        if save_video:
            out_video.release()
            log("Output has been save to {}".format(out_video_path))


if __name__ == '__main__':
    main()
