<p align="center"><img width="25%" src="./utils/materials/kalray_logo.png"></a></br></p>

# WIKI - KaNN™ Model Zoo

![ACE-6.1.0](https://img.shields.io/badge/MMPA--Coolidge2-ACE--6.1.0-g) ![KaNN-5.6.0](https://img.shields.io/badge/KaNN--5.6.0-red)
![Classification](https://img.shields.io/badge/Classification-27-blue) ![Object-Detection](https://img.shields.io/badge/Object%20Detection-41-blue) ![Segmentation](https://img.shields.io/badge/Segmentation-9-blue)</br>

The Kalray KaNN™ Model Zoo repository offers a collection of neural network models **ready to compile & run** on Kalray's MPPA®
manycore processor. Coolidge V2, the 3rd and latest generation of our MPPA®, is a dedicated processor for **AI applications**.
KaNN™ Model Zoo complements the KaNN™ SDK, which streamlines model generation and optimizes **AI performance** on Kalray's processors.

## Table of contents

- [WIKI - KaNN™ Model Zoo](#wiki---kann-model-zoo)
  - [Table of contents](#table-of-contents)
  - [KaNN™ framework description](#kann-framework-description)
  - [Prerequisites: SW environment \& configuration](#prerequisites-sw-environment--configuration)
  - [How models are packaged](#how-models-are-packaged)
  - [Generate a model to run on the MPPA®](#generate-a-model-to-run-on-the-mppa)
  - [Evaluate the neural network inference on the MPPA®](#evaluate-the-neural-network-inference-on-the-mppa)
  - [Run the neural network in a video pipeline](#run-the-neural-network-in-a-video-pipeline)
  - [Neural networks accuracy and associated metrics](#neural-networks-accuracy-and-associated-metrics)
    - [Definitions](#definitions)
    - [How metrics are computed](#how-metrics-are-computed)
    - [Steps to evaluate the accuracy of a generated neural network (object-detection)](#steps-to-evaluate-the-accuracy-of-a-generated-neural-network-object-detection)
  - [Custom Layers for extended neural network support](#custom-layers-for-extended-neural-network-support)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Automated tests, benchmark](#automated-tests-benchmark)

## KaNN™ framework description

<img width="100%" src="./utils/materials/kann_process.png"></a></br>

Kalray Neural Network (KaNN™) is a SDK included in the AccessCore Embedded (ACE) compute
 offer to optimize AI inference on MPPA®.
 
 The SDK leverages the possibility to parse and analyze a Neural Network Graph from the standardized ONNX
 framework for model interoperability and generate MPPA® byte code to achieve best performance efficiency.
 This wiki does not contain any information about the use of the KaNN™ API, but it helps to deploy and
 evaluate quickly AI solutions from preconfigured and validated model.

 For details, please do not hesitate to read the documentation 😏, contact us directly at
 support@kalrayinc.com or report an issue at https://github.com/kalray/kann-model-zoo/issues.

> [!TIP]
> To deploy your solution from an identified neural network, the steps are all easy 😃:
>
> 1. From an identified Neural Network, generate the KaNN™ model bytecode (no HW dependencies)
>
>    ```bash
>    kann generate --model=Kalray/mnist/mnist.onnx -d kMnist
>    ```
> 2. Run the model from application:
>
>    ```bash
>    kann run kMnist
>    ```

NB: Running the model requires a PCIe board with Kalray's MPPA®. See our product [here](README.md#hardware-requirements)

## Prerequisites: SW environment & configuration

Source the Kalray's AccessCore® environment, at the following location:
```bash
 source /opt/kalray/accesscore/kalray.sh
 ```
and check the envrionment variable `$KALRAY_TOOLCHAIN_DIR` is not empty.

If it does not exist, please configure a specific virtual python environment (recommended)
*in a location of your choice*. E.g.:
```bash
export KANN_ENV=$HOME/.local/share/python3-kann-venv
python3 -m venv $KANN_ENV
```
Source your python environment:
```bash 
source $KANN_ENV/bin/activate
```

Install locally the KaNN™ wheel and all dependencies (it supposed ACE Release is installed
 in `$HOME` directory):
```bash
pip install $HOME/ACE6.1.0-SDK/packages/python/kann-5.6.0-py3-none-any.whl
```

Lastly, do the same for additional python requirements of this repo:
```bash 
pip install -r requirements.txt
```
Please also refer to the ACE SDK install procedure detailed [here](https://lounge.kalrayinc.com/hc/en-us/articles/20877509597084-ACE-6-1-0-Content-installation-release-note-and-Getting-Started-Coolidge-v2)

You are now all set up use the KaNN™ Model Zoo. Please don't forget to source **your** python
environment any time you open a new terminal or adding the following lines to your .bashrc,
.bashalias, or similar according to *your* choice earlier.
```bash
export KANN_ENV=$HOME/.local/share/python3-kann-venv
source /opt/kalray/accesscore/kalray.sh
source $KANN_ENV/bin/activate
```

## How models are packaged

Each model is packaged to be compiled and run with KaNN™ SDK. In each model directory, you'll find:
- a pre-processing python script: `input_preparator.py`
- a post-processing directory: `output_preparator/`
- a model dir (empty), model wil be downloaded once the model is called for generation
- configuration files (*.yaml) for generation:
    * `network_f16.yaml` :  batch 1 - FP16 - nominal performance
    * `network_i8.yaml` :   batch 1 - FP16/Q-INT8 - nominal performance

Models LICENSE and SOURCES are described individually in our HuggingFace space, available at
https://huggingface.co/Kalray.

<p align="center">
  <img width="25%" alignment="center" src="./utils/materials/Hugging_Face_logo.svg"></a></br>
</p>

## Generate a model to run on the MPPA®

Use the following command to generate an model to run on the MPPA®:
```bash
# syntax: kann generate <configuration_file.yaml> -d <generated_path_dir>
kann generate networks/object-detection/yolov8/onnx/yolov8n_f16.yaml -d yolov8n
```

It will provide you into the path directory `generated_path_dir`, here called `yolov8n`:
* a <my_network>.kann file (network contents with runtime and context information)
* a network.dump.yaml file (a copy of the configuration file used)
* a log file of the generation

Please refer to Kalray's documentation and KaNN user manual provided for more details
to fine-tune (or optimize) the inference.

From the generate command, you are able to list the available models locally:
```bash
./generate --list
```

From the list printed in the terminal, it is now possible to generate a neural
 network to execute on the MPPA®, for example :
```bash
./generate -n yolov8 -d yolov8n
```

## Evaluate the neural network inference on the MPPA®

Kalray's toolchain integrates its own host application named `kann_opencl_cnn` to run compiled models.
To evaluate the performance of the neural network on the MPPA®, two methods :
  + use `./run` script wit the `infer` sub-command (it will use the `kann run` command indirectly)
  + or directly, `kann run` cli-commmand offered by KaNN

Use the following command to start quickly the inference:
```bash
# syntax: kann run <generated_path_dir>#
kann run yolov8n
```
or
```bash
# syntax: ./run infer <generated_path_dir>
./run infer yolov8n
```

> [!TIP]
> From ACE >= 6.1.0, KaNN generates directly an ONNX model from HuggingFace with this method.
> Do not hesitate to generate and run a model quickly with the following manner:

```bash
# $ kann run --model=<HF_REPO_ID>/<FILENAME.onnx>
kann run --model=Kalray/yolov8/yolov8n.onnx
```

## Run the neural network in a video pipeline

Use the command below to start the inference of the model, just
generated, into a video pipeline. It will include the inference with a pre-
and post-processing scripts with a video/image stream input, supported by
the OpenCV Python API.

```bash
# syntax : ./run demo <generated_path_dir> <source_file_path>
./run demo yolov8n ./utils/sources/cat.jpg
```

> [!CAUTION]
> `./run demo` is a wrapper to `./utils/kann_video_demo.py` python script.
> Please, consider that:
> 1. This script is a demonstrator to include a pre-processing and
>    post-processing with the inference of a model on the MPPA.
> 2. This script is dedicated to a **HOST+k300-board** environment, but
>    processes are not masked. It means that performance is not efficient.
> 3. Using **TurboCard4**, it would be functional, but 1 pipeline is dedicated to
>    1 MPPA. So, 4 streams on TC4 need to be opened to use all processors available.
>    NB: PCIe time transfer is decreased to x4 instead x16 lanes in PCIe-GEN4.

All timings are logged by the video demo script, and reported such as:
+ read : time to import frame
+ pre  : pre processing time
+ send : copy data to FIFO in
+ kann : wait until the FIFO out is filled (including the neural network inference)
+ post : post processing time
+ draw : draw annotation on input frame
+ show : time to display the image though opencv
+ total: sum of the previous timings

> [!TIP]
> Interested to run faster ? please contact our support to optimize your use case at support@kalrayinc.com

To disable the L2 cache at runtime add the `--l2-off` argument (automatically detected):
```bash
./run --l2-off demo yolov8n ./utils/sources/dog.jpg
```
This allows using a larger fraction of the MPPA®'s DDR for data buffers.
 Disabling L2 cache is also implicitly done in KaNN™ Model Zoo if we detect the 
 `data_buffer_size` in the model's configuraiton `*.yaml` file requires us to do so.
 A warning will be displayed if L2 cache is disabled without explicitly setting the
 flag as mentioned above.

Please find below some interesting commands:

* Disable the display (automatically detected):
```bash
./run demo yolov8n ./utils/sources/street/street_0.jpg --no-display
```
* Disable the replay (for a video or an image):
```bash
./run demo yolov8n ./utils/sources/street/street_0.jpg --no-replay
```
* Save the last frame annotated into the current dir:
```bash
./run demo yolov8n ./utils/sources/street/street_0.jpg --no-replay --save-img --verbose
```
* Iterate on the 5 first frames:
```bash
./run demo --device=cpu yolov8n ./utils/sources/street/street_0.jpg -n 5 --save-img --verbose
```
* Run on the CPU target (in order to compare results):
```bash
./run demo --device=cpu yolov8n ./utils/sources/street/street_0.jpg -n 1 --save-img --verbose
```

> [!TIP]
> Do not hesitate to use the keyword `--help` to display options


Demonstration scripts are provided in python.

> [!CAUTION]
> `kann_opencl_cnn` is a simple and generic host application for
> neural network inference on MPPA®. It does not use pipelining. Thus video pipeline
> is **NOT FULLY OPTIMIZED** and  requires custom developments to benefit of the full
> performance of the MPPA®, depending of your own environment and system. Do not
> hesitate to contact our services <support@kalrayinc.com> to optimize your custom AI solution.

Please take a look to our notebooks included in the repository (see [Jupyter Notebooks](#jupyter-notebooks))

## Neural networks accuracy and associated metrics

### Definitions

In this repository, neural networks can predict frames by :

* classification label ID (classifiers),
* one or multiple bounding box(es) to point to an object (object-detection)
* masks associated to an object (segmentation)

To consider the accuary of a prediction, some metrics needs to be defined here such as:

* **Precision**: True Positives (TP) ratio with the sum of TP and False Positives (FP)
* **Recall**: True Positives (TP) ratio with the sum of TP and False Negatives (FN)
* **F1-score**: The harmonic mean of the precision (P) and recall (R)

Details can be found on scikit-learn documentation : https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html or on wikipedia directly: https://en.wikipedia.org/wiki/Precision_and_recall

* **IoU**: ratio between the area where the predicted bbox and the ground-truth bbox overlap over the total area covered the both.
* **mAP50**: calculated using a fixed IoU threshold of 0.5.
* **mAP50-95**: calculated by averaging the mAP across multiple IoU thresholds, typically 10 points from 0.5 to 0.95.

as explained : https://www.ultralytics.com/glossary/mean-average-precision-map

### How metrics are computed

Considering that defintions :
* True positive (TP): Number of instances predicted as `True` and is really `True`
* False positive (FP): Number of instances predicted as `True` and must be `False`
* False negative (FN): Number of instances predicted as `False` and would be `True`

The metrics are computed as below :
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1-score**: `2*TP / (2*TP + FP + FN)`

*For classifiers* : we are using the common metric "Top-k Accuracy classification score", defined here https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html
  - **Top-1 acc**: This metric computes the number of times where the correct label is among the top 1 labels predicted.
  - **Top-5 acc**: This metric computes the number of times where the correct label is among the top 5 labels predicted.

*For object-detection* :
- **mAP50**: mean average precision on a IoU threshold of 0.5
- **mAP50-95**: mean average precision on a IoU threshold with 10 points from 0.5 to 0.95

*For segmentation* :
- **mIoU acc**: mean Intersection over union, the metric is defined by the overlap between the predicted mask segmentation and the ground truth, divided by the total area covered by the union of the two
- **mAP50-95 mask**: mean average precision on a mean mask IoU threshold with 10 points from 0.5 to 0.95

### Steps to evaluate the accuracy of a generated neural network (object-detection)

NB: This section is illustrated by the jupyter notebook at [eval_map.ipynb](./notebooks/eval_map.ipynb)

From this repository, use one of our packaged model, e.g **YOLOv8s**. Then follow each step below:

1. Generate the target neural network

```bash
kann generate ./networks/object-detection/yolov8/onnx/yolov8s_f16.yaml -d yolov8s
```

2. Ensure that computation of image pipepline prints the data, label ID and bounding box in verbose mode

```bash
./run demo yolov8s utils/sources/cat.jpg --no-replay --save-img --verbose
```
> [!IMPORTANT]
> the script `evaluate` would trig on the output of the post-processing script
> when the network detects something, the syntax below can be found as *(conf, label ID, bbox[x1, y1, x2, y2])*:
> `>> [Post-proc] prediction: 0.8567 - cat - [109.09914, 47.521957, 428.05875, 464.4724]`

3. Finally, execute the script `evaluate` with the dataset `coco` (5k images), `coco8` (4 images) or `coco128` (128 images)

```bash
./evaluate yolov8s --metrics=mAP --dataset=coco8 --all
...
INFO: Dataset coco8 downloaded and extracted to /work1/qmuller/kann-model-zoo/utils/datasets/coco8/coco8
INFO: Processing images in [1] steps of nb-images : 4
INFO: STEP 1 / 1
Preparing images for inference : 100%|██████████████████████| 4/4 [00:00<00:00, 37.79it/s]
INFO: Running inference on MPPA ...
Post-processing predictions 1/1: 100%|█████████████████████| 4/4 [00:00<00:00, 285.27it/s]
INFO: STATISTICS:
INFO:                  Class     Images  Instances       Prec     Recall   F1-score      mAP50   mAP50-95
INFO:                    all          4         17      0.864      0.921      0.883      0.924      0.723
INFO:                 person          3         10       0.79        0.6      0.682      0.735      0.386
INFO:                    dog          1          1          1          1          1      0.995      0.796
INFO:                  horse          1          2      0.748          1      0.856      0.995      0.771
INFO:               elephant          1          2      0.643      0.929       0.76      0.828      0.493
INFO:               umbrella          1          1          1          1          1      0.995      0.995
INFO:           potted_plant          1          1          1          1          1      0.995      0.895
INFO: 
INFO: Evaluation time on COCO8 takes 2.186 secs.
```

The neural network mAP50-95 is evaluated at **71.7%** and would be less more. This is due to the lack of the
number of images. Typically, the COCO evaluation dataset is 5K images and final prediction accuracy converges to
a final results close to the trained model value. COCO128 is a good aproximation to provide a quick and
correct evaluation.

To compare with an alternative solution, for example with Ultralytics framework, follow these steps :
```bash
pip install ultralytics
yolo export model=yolov8s.pt format=onnx batch=1 imgsz=640
yolo val model=yolov8s.onnx data=coco8.yaml batch=1 imgsz=640
# returns
       Class     Images  Instances      Box(P          R      mAP50   mAP50-95):
         all          4         17      0.813      0.832      0.917      0.714
      person          3         10          1      0.492      0.693      0.377
         dog          1          1      0.932          1      0.995      0.796
       horse          1          2      0.844          1      0.995        0.8
    elephant          1          2      0.621        0.5      0.828      0.418
    umbrella          1          1      0.731          1      0.995      0.995
potted plant          1          1      0.751          1      0.995      0.895
```

## Custom Layers for extended neural network support

According to the Kalray's documentation in KaNN™ manual, users have the possibility to integrate
custom layers in case it is not supported by KaNN™. This can be done by following these
general steps:
1. Implement the python function callback to ensure that KaNN™ generator is able to support the layer
2. Implement the layer python class to ensure that arguments are matching with the C-function
3. Implement the C-function into the SimpleMapping macro, provided in the example
4. Build the C-function with Kalray makefile and reuse it for inference

For more details please refer to the KaNN™ user manual.

To ensure extended support of all neural networks provided in the repository (such as YOLOv8) the
 `kann_custom_layers` directory contains implementations of the following custom layers:
 * SiLU

Please follow these few steps to use custom layer implementations, for example to support YOLOv8:
1. Configure your software environment:
```bash
KANN_ENV=$HOME/.local/share/python3-kann-venv
source /opt/kalray/accesscore/kalray.sh
source $KANN_ENV/bin/activate
```
2. Add "Silu" callback to add your custom layer in the compiler (from `./kann_custom_layers/kann_custom_layers.py:L112`):
```python
# change
onnx_parser_callbacks = {
    'Gather': onnx_gather_callback
}
# to
onnx_parser_callbacks = {
    'Silu': onnx_silu_parser_callback
}
# or just add the 'Silu' keys to onnx_parser_callbacks dict
```
3. Then, build custom kernels to run over the MPPA®:
```bash
make -BC kann_custom_layers O=$PWD/output
```
4. Generate the model:
```bash
PYTHONPATH=$PWD/kann_custom_layers kann generate $PWD/networks/object-detection/yolov8/onnx/yolov8n_f16.yaml -d yolov8n-custom
```
During generation this point would be highlighted:
```
[INFO]  Found kann_custom_layers with callbacks
[INFO]  - Silu
[INFO]  - Gather
[INFO]  ---
```
Finally, in the STATIC log, the custom-kernel associated to your Custom-Layer is shown:
```
silu_x8_tf16_tf16 :    57[108.9 M] |    57[113.9 M] |    57[118.1 M] |    57[113.9 M] |    57[114.7 M]
```
5. Run demo with generated the generated directory (`yolov8n-custom` in this example) and the newly compiled kernels (.pocl file) for the MPPA®:
```bash
./run --pocl-dir=$PWD/output/opencl_kernels demo --device=mppa yolov8n-custom ./utils/sources/cat.jpg --verbose
```
or run the model on CPU target (in order to compare results):
```bash
./run demo --device=cpu yolov8n-custom ./utils/sources/cat.jpg --verbose
```

## Jupyter Notebooks

You may also notice a folder called `./notebooks/`  which is available in this repository.
It provides additional usage examples. Let's take a look at:
* [x] [Quick Start](./notebooks/quick_start.ipynb): Generate and run a neural network from the KaNN™ Model Zoo
* [x] [Accuracy](./eval_map.ipynb): Evaluate the mAP50 of an object-detection model

To execute it, please set up your python environment and be sure you could use correctly your preferred web browser
(firefox, google-chrome, ... for example) :

```bash
# source YOUR python environment if not done
KANN_ENV=$HOME/.local/share/python3-kann-venv
source $KANN_ENV/bin/activate
# install jupyter notebook package
pip install jupyter
# wait that all dependencies are installed ...
```

From kann-model-zoo home directory, then open the desired notebook:

```bash
jupyter notebook notebooks/quick_start.ipynb &
```

A new window will appear such as

<img width="100%" src="./utils/materials/jupyter_notebooks.png"></a></br>

Finally, select & click to `Run` > `Run All Cells (Shift+Enter)` to execute all commands in-line ...
et voilà 😃. Don't forget restart the kernel if needed and to kill the jupyter notebook server once you're done.


## Automated tests, benchmark

From this repository, an automated script can run all models available in kann-model-zoo.

> [!CAUTION]
> All models will be executed WITHOUT fine tuned parameters (in YAML file).
> Please consider that final performance reported is not the best you could reach.
> However, it would give you a first idea of the MPPA performance, its functionality and accuracy

> [!TIP]
> Interested to run faster ? please contact our support to optimize your use case at support@kalrayinc.com

The command to run all models in this reposotry is:
```
python3 valid/jenkins/unit_tests.py --build-dir=./build
```

Finally a CSV report is available at `./build/logs/report/report.csv` as shown below (as MD table):

| test_id       | family | name                | framework | dtype | cfg_file                                                                | gen_status | infer_status | host_fps | mppa_fps | cycles    | demo_mppa_status | score_mppa | pred_mppa                  | bbox_mppa | demo_cpu_status | score_cpu | pred_cpu                   | bbox_cpu | accuracy |
| ------------- | ------ | ------------------- | --------- | ----- | ----------------------------------------------------------------------- | ---------- | ------------ | -------- | -------- | --------- | ---------------- | ---------- | -------------------------- | --------- | --------------- | --------- | -------------------------- | -------- | -------- |
| class-0000f16 | class  | alexnet_f16         | ONNX      | fp16  | /***/networks/classifiers/alexnet/onnx/alexnet_f16.yaml                 | True       | True         | 221.3    | 227.9    | 4,387,226 | True             | 0.49       | n02123159 tiger cat        | None      | True            | 0.49      | n02123159 tiger cat        | None     | 1.0      |
| class-0001f16 | class  | densenet_121_f16    | ONNX      | fp16  | /***/networks/classifiers/densenet-121/onnx/densenet_121_f16.yaml       | True       | True         | 252.0    | 259.7    | 3,850,302 | True             | 0.385      | n02123045 tabby, tabby cat | None      | True            | 0.384     | n02123045 tabby, tabby cat | None     | 0.997    |
| class-0002f16 | class  | densenet_169_f16    | ONNX      | fp16  | /***/networks/classifiers/densenet-169/onnx/densenet_169_f16.yaml       | True       | True         | 182.6    | 187.5    | 5,330,580 | True             | 0.529      | n02123045 tabby, tabby cat | None      | True            | 0.528     | n02123045 tabby, tabby cat | None     | 0.998    |
| class-0003f16 | class  | efficientnet_b0_f16 | ONNX      | fp16  | /***/networks/classifiers/efficientnet-b0/onnx/efficientnet_b0_f16.yaml | True       | True         | 154.8    | 160.1    | 6,244,827 | True             | 0.332      | n02123045 tabby, tabby cat | None      | True            | 0.334     | n02123045 tabby, tabby cat | None     | 0.994    |
...

Otherwise, you could focus on particular networks, such as:
```
python3 valid/jenkins/unit_tests.py -c object-detection -i yolov5 --build-dir=./build
```

Authors:
 + Quentin Muller <qmuller@kalrayinc.com>
 + Björn Striebing <bstriebing@kalrayinc.com>
