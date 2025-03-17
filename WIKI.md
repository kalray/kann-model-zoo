# WIKI - KaNN™ Model Zoo

<img width="30%" src="./utils/materials/mppa-processor.jpg"></a></br>

![ACE-6.0.0](https://img.shields.io/badge/MMPA--Coolidge2-ACE--6.0.0-g) ![KaNN-5.5.0](https://img.shields.io/badge/KaNN--5.5.0-red)
![Classification](https://img.shields.io/badge/Classification-29-blue) ![Object-Detection](https://img.shields.io/badge/Object%20Detection-32-blue) ![Segmentation](https://img.shields.io/badge/Segmentation-9-blue)</br>

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
  - [Run the neural network as a demo](#run-the-neural-network-as-a-demo)
  - [Neural networks accuracy and associated metrics](#neural-networks-accuracy-and-associated-metrics)
    - [Definitions](#definitions)
    - [How metrics are computed](#how-metrics-are-computed)
    - [Steps to evaluate the accuracy of a generated neural network (object-detection)](#steps-to-evaluate-the-accuracy-of-a-generated-neural-network-object-detection)
  - [Custom Layers for extended neural network supoort](#custom-layers-for-extended-neural-network-supoort)
  - [Jupyter Notebooks](#jupyter-notebooks)


## KaNN™ framework description

<img width="500" src="./utils/materials/CNN.png"></a></br>

Kalray Neural Network (KaNN™) is a SDK included in the AccessCore Embedded (ACE) compute
 offer to optimize AI inference on MPPA®.
 
 It leverages the possibility to parse and analyze a Convolutional Neural Network (figure above)
 from the standardized ONNX framework for model interoperability and generate MPPA® byte code to
 achieve best performance efficiency. This wiki does not contain any information about the use of the
 KaNN™ API, but it helps to deploy and quickly evaluate AI solutions from preconfigured and validated model. 
 For details, please do not hesitate to read the documentation 😏 or contact us directly at support@kalrayinc.com

So, to deploy your solution from an identified neural network, the steps are all easy 😃:

1. From a CNN, generate the KaNN™ model bytecode (no HW dependencies)

   ```bash
   kann generate --model=Kalray/mnist/mnist.onnx -d kMnist
   ```
2. Run the model from demo application (Python script & host application are included in the repository
   and AccessCore software)

   ```bash
   kann run kMnist
   ```

NB: Running the model requires a PCIe board with Kalray's MPPA®.

## Prerequisites: SW environment & configuration

Source the Kalray's AccessCore® environment, at the following location:
```bash
 source /opt/kalray/accesscore/kalray.sh
 ```
and check the envrionment variable `$KALRAY_TOOLCHAIN_DIR` is not empty.

If it does not exist, please configure a specific virtual python environment (recommended) *in a location of your choice*. E.g.:
```bash
KANN_ENV=$HOME/.local/share/python3-kann-venv
python3 -m venv $KANN_ENV
```
Source your python environment:
```bash 
source $KANN_ENV/bin/activate
```

Install locally the KaNN™ wheel and all dependencies (it supposed the ACE Release is installed in `$HOME` directory):
```bash
pip install $HOME/ACE6.0.0/KaNN-generator/kann-*.whl
```

Lastly, do the same for additional python requirements of this repo:
```bash 
pip install -r requirements.txt
```
Please also refer to the ACE SDK install procedure detailed [here](https://lounge.kalrayinc.com/hc/en-us/articles/19422051062940-ACE-6-0-0-Content-installation-release-note-and-Getting-Started-Coolidge-v2)

You are now all set up use the KaNN™ Model Zoo. Please don't forget to source **your** python environment any time you open a new terminal or adding the following lines to your .bashrc, .bashalias, or similar according to *your* choice earlier.
```bash
KANN_ENV=$HOME/.local/share/python3-kann-venv
source /opt/kalray/accesscore/kalray.sh
source $KANN_ENV/bin/activate
```

## How models are packaged

Each model is packaged to be compiled and run with KaNN™ SDK. In each model directory, you'll find:
- a pre-processing python script: `input_preparator.py`
- a post-processing directory: `output_preparator/`
- a model dir (empty), model wil be downloaded once the model is called for generation
- configuration files (*.yaml) for generation:
    * network_f16.yaml :  batch 1 - FP16 - nominal performance
    * network_i8.yaml :   batch 1 - FP16/Q-INT8 - nominal performance

Models LICENSE and SOURCES are described individually in our HuggingFace space, available at
https://huggingface.co/Kalray.

<p align="center">
  <img width="25%" alignment="center" src="./utils/materials/Hugging_Face_logo.svg"></a></br>
</p>

## Generate a model to run on the MPPA®

Use the following command to generate an model to run on the MPPA®:
```bash
# syntax: ./generate -c <configuration_file.yaml> -d <generated_path_dir>
./generate -c networks/object-detection/yolov8n-relu/onnx/network_f16.yaml -d yolov8n
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
./generate -n yolov8n
```

## Evaluate the neural network inference on the MPPA®

Kalray's toolchain integrates its own host application named `kann_opencl_cnn` to run compiled models.
To evaluate the performance of the neural network on the MPPA®, two methods :
  + use `./run` script wit the `infer` sub-command (it will use the `kann run` command indirectly)
  + or directly, `kann run` cli-commmand offered by KaNN

Use the following command to start quickly the inference:
```bash
# syntax: ./run infer <generated_path_dir>
./run infer yolov8n
```
or
```bash
# syntax: kann run <generated_path_dir> --nb-frames=25
kann run yolov8n -n 25
```

**NOTES 😎**

Now, with kann it is possible to evaluate directly an ONNX model from HuggingFace with this method:
```bash
# $ kann generate --model=<HF_REPO_ID>/<FILENAME.onnx> -d <GEN_DIR>
# $ kann run <GEN_DIR>
kann generate --model=Kalray/yolov8n-relu/yolov8n-relu-s.optimized.onnx \
  --quantize_fp32_to_fp16=True -d yolov8n
# then
kann run yolov8n
```


## Run the neural network as a demo

Use the following command to start the inference of, the model just
generated, from a video pipeline. It will include the inference into a pre-
and post-processing scripts with a video/image stream input, supported by
the OpenCV Python API.

```bash
# ./run demo <generated_path_dir> <source_file_path>
./run demo yolov8n ./utils/sources/cat.jpg
```

All timings are logged by the video demo script, and reported such as:
+ read : time to import frame
+ pre  : pre processing time
+ send : copy data to FIFO in
+ kann : wait until the FIFO out is filled (including the neural network inference)
+ post : post processing time
+ draw : draw annotation on input frame
+ show : time to display the image though opencv
+ total: sum of the previous timings

To disable the L2 cache at runtime add the '--l2-off' argument:
```bash
./run --l2-off demo yolov8n ./utils/sources/dog.jpg
```
This allows using a larger fraction of the MPPA®'s DDR for data buffers.
 Disabling L2 cache is also implicitly done in KaNN™ Model Zoo if we detect the 
 `data_buffer_size` in the model's configuraiton `*.yaml` file requires us to do so.
 A warning will be displayed if L2 cache is disabled without explicitly setting the
 flag as mentioned above.

To disable the display:
```bash
./run demo yolov8n ./utils/sources/street/street_0.jpg --no-display
```

To disable the replay (for a video or a image):
```bash
./run demo yolov8n ./utils/sources/street/street_0.jpg --no-replay
```

Save the last frame annotated into the current dir:
```bash
./run demo yolov8n ./utils/sources/street/street_0.jpg --no-replay --save-img --verbose
```

To run on the CPU target (in order to compare results):
```bash
./run demo --device=cpu yolov8n ./utils/sources/street/street_0.jpg --no-replay --save-img --verbose
```

Demonstration scripts are provided in python.

> **PLEASE NOTE**
> `kann_opencl_cnn` is a simple and generic host application for neural network inference on MPPA®.
> It does not use pipelining. Thus video pipeline is **NOT FULLY OPTIMIZED** and  requires custom developments to 
> benefit of the full performance of the MPPA®, depending of your own environment and system. Do not hesitate to 
> contact our services <support@kalrayinc.com> to optimize your solution.

Please take a look to our notebooks included in the repository (see [Jupyter Notebooks](#jupyter-notebooks))


## Neural networks accuracy and associated metrics

### Definitions

In this repository, neural networks can predict in one image/frame:

* a classification label ID (classifiers),
* one or multiple bounding box(es) to point to an object (object-detection)
* a mask associated to an object (segmentation)

To consider the accuary of a prediction, some metrics needs to be defined here such as:

* **Precision**: True Positives (TP) ratio with the sum of TP and False Positives (FP)
* **Recall**: True Positives (TP) ratio with the sum of TP and False Negatives (FN)
* **F1-score**: The harmonic mean of the precision (P) and recall (R)

Details can be found on scikit-learn documentation here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html or on wikipedia directly: https://en.wikipedia.org/wiki/Precision_and_recall

* **IoU**: ratio between the area where the predicted bbox and the ground-truth bbox overlap over the total area covered the both.
* **mAP50**: calculated using a fixed IoU threshold of 0.5.
* **mAP50-95**: calculated by averaging the mAP across multiple IoU thresholds, typically 10 points from 0.5 to 0.95.

as explained here: https://www.ultralytics.com/glossary/mean-average-precision-map

### How metrics are computed

Considering that defintions :
* True positive (TP): Number of instances predicted as `True` and is really `True`
* False positive (FP): Number of instances predicted as `True` and must be `False`
* False negative (FN): Number of instances predicted as `False` and would be `True`

The metrics are computed as below :
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1-score**: `2*TP / (2*TP + FP + FN)`

*For classifiers* :
- we are using the common metric "Top-k Accuracy classification score", defined here https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html
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
./generate networks/object-detection/yolov8s/onnx/network_f16.yaml -d yolov8s
```

2. Ensure that input and output preparator are included
```bash
ls -R yolov8s/
```

3. Ensure that computation of image pipepline prints the data, label ID and bounding box in verbose mode
```bash
./run demo yolov8s utils/sources/cat.jpg --no-replay --save-img --verbose
```
**IMPORTANT NOTE**:
> the script `eval` would trig on the output of the post-processing script
> when the network detects something, the syntax below can be found *(conf, label ID, bbox[x1, y1, x2, y2])*:
> `>> [Post-proc] prediction: 0.68 - cat - [113, 47, 351, 462]`

4. Finally, execute the script `eval` with the dataset `coco` (5k images), `coco8` (4 images) or `coco128` (128 images)
```bash
./eval yolov8s --metrics=mAP --dataset=coco8
INFO: Processing images in steps of nb-images : [4]
Preparing images for inference: 100%|███████████████████████| 4/4 [00:00<00:00, 269.67it/s]
INFO: Running inference on MPPA 100%|███████████████████████| 4/4 [00:00<00:00
Post-processing predictions 1/1:100%|███████████████████████| 4/4 [00:00<00:00, 476.08it/s]
Processing pipeline :           100%|███████████████████████| 1/1 [00:01<00:00,  1.95s/it]
INFO:
INFO:                  Class     Images  Instances       Prec     Recall   F1-score      mAP50   mAP50-95
INFO:                    all          4         17      0.833      0.733      0.762       0.78      0.598
INFO:                 person          3         10          1        0.4      0.571        0.7      0.314
INFO:                    dog          1          1          1          1          1      0.995      0.796
INFO:                  horse          1          2          1          1          1      0.995      0.784
INFO:               elephant          1          2          0          0          0          0          0
INFO:               umbrella          1          1          1          1          1      0.995      0.796
INFO:           potted_plant          1          1          1          1          1      0.995      0.895
```

The neural network mAP50-95 is here evaluated at **59.8%** and would be less more. This is due to the lack of the
number of images. Typically, the COCO evaluation dataset is 5K images and final prediction accuracy converges to 
a final results close to the trained model value. COCO128 is a good aproximation to provide a correct evaluation.

Do not hesitate to compare with ultralytics, following these steps, for example with Ultralytics framework:
```bash
pip install ultralytics
yolo export model=yolov8s.pt format=onnx batch=1 imgsz=640
yolo val model=yolov8s.onnx data=coco8.yaml batch=1 imgsz=640
```

## Custom Layers for extended neural network supoort

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

2. Then, build custom kernels to run over the MPPA®:
```bash
make -BC kann_custom_layers O=$PWD/output
```

3. Generate the model:
```bash
PYTHONPATH=$PWD/kann_custom_layers ./generate $PWD/networks/object-detection/yolov8n/onnx/network_f16.yaml -d yolov8n
```

4. Run demo with generated the generated directory (`yolov8n` in this example) and the newly complied kernels (.pocl file) for the MPPA®:
```bash
./run --pocl-dir=$PWD/output/opencl_kernels demo --device=mppa yolov8n-custom ./utils/sources/cat.jpg --verbose
```
or run the model on CPU target (in order to compare results):
```bash
./run demo --device=cpu yolov8n-custom ./utils/sources/cat.jpg --verbose
```

## Jupyter Notebooks

You may also notice a folder called `./notebooks/`  which is available in this repository. It provides additional usage examples. Let's take a look at:
* [x] [Quick Start](./notebooks/quick_start.ipynb): Generate and run a neural network from the KaNN™ Model Zoo
* [x] [Evaluate the mAP of an object-detection neural network ](./eval_map.ipynb)

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

<img width="100%" src="./utils/materials/quick_start_notebook.png"></a></br>

Finally, select & click to `Run` > `Run All Cells (Shift+Enter)` to execute all commands in-line ...
et voilà 😃. Don't forget restart the kernel if needed and to kill the jupyter notebook server once you're done.

Other notebooks will be soon available:
* [ ] Advanced: import a neural network and create a package to run on the MPPA
* [ ] Graph inspection: analyze a neural network generated by kann
* [ ] Fine-tune: optimize the generation of a neural network
* [ ] Custom layer (basic) : use the custom layer already implemented in this repository
* [ ] Custom layer (advanced): Implement a custom layer to support a specific network
* [ ] Custom kernel (advanced): Implement a custom kernel to support a specific network
* [ ] Custom kernel (expert): Optimie a custom kernel to accelerate a specific network


Authors:
 + Quentin Muller <qmuller@kalrayinc.com>
 + Björn Striebing <bstriebing@kalrayinc.com>
