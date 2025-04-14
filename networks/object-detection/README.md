<img width="20%" src="../../utils/materials/kalray_logo.png"></a>

## List of Object Detection Neural Networks
This repository gives access to following object dectection neural networks main architecture:
* EfficientDet, RetinatNet, SSD, YOLO

## Important notes

* Neural networks are available on our **Hugging face plateform** 🤗 [HERE](https://huggingface.co/Kalray).
  Do not hesitate to check model cards for details of implementation, sources and/or license.

* All models have been trained on: **[COCO2017](https://cocodataset.org/#detection-2017) dataset**

* To generate a neural network with KaNN :
  + in FP16, please refer to ONNX model (pointed by network_f16.yaml)
  + Please refer to [WIKI.md](../../WIKI.md) for instructions on how to use any of these models

## Neural Networks

The models are listed below, according:
  + The accuracy metrics (mAP50 and mAP50/95)
  + MPPA performance at **batch 1** in :
    * Frame per seconds (according that MPPA frequency is 1.0GHz along inference)
    * Total number of cycles to compute 1 frames (averaged on 10 first frames)
    * K Floating operations (FLOPS) per machine cycles (c)

*NB: MPPA Coolidge V2 processor default frequency is 1.0 GHz*

<!-- START AUTOMATED TABLE -->
| NAME                                                                             |   FLOPs | Params | mAP-50 | mAP-50/95 | Framework |   Input   | 🤗 HF repo-id                                                                                  | FPS(MPPA) | TOTAL(Mc) | KFLOPS/c |
| :------------------------------------------------------------------------------- | ------: | -----: | :----: | :-------: | :-------: | :-------: | :-------------------------------------------------------------------------------------------- | --------: | --------: | -------: |
| [EfficientDet-D0](./efficientdet-d0/onnx/network_f16.yaml)                       |  10.2 G |  3.9 M |   -    |  53.0 %   |   ONNX    |  512x512  | [Kalray/efficientdet-d0](https://huggingface.co/Kalray/efficientdet-d0)                       |      37.3 |     26.75 |    0.368 |
| [FasterRCNN-resnet50](./faster-rcnn-rn50/onnx/network_f16.yaml)                  |  94.1 G | 26.8 M |   -    |  35.0 %   |   ONNX    |  512x512  | [Kalray/faster-rcnn-rn50](https://huggingface.co/Kalray/faster-rcnn-rn50)                     |      65.6 |     15.32 |    6.145 |
| [RetinaNet-resnet101](./retinanet-resnet101/onnx/network_f16.yaml)               | 161.4 G | 56.9 M |   -    |     -     |   ONNX    |  512x512  | [Kalray/retinanet-resnet101](https://huggingface.co/Kalray/retinanet-resnet101)               |      39.8 |     25.06 |    6.410 |
| [RetinaNet-resnet50](./retinanet-resnet50/onnx/network_f16.yaml)                 | 122.4 G | 37.9 M |   -    |     -     |   ONNX    |  512x512  | [Kalray/retinanet-resnet50](https://huggingface.co/Kalray/retinanet-resnet50)                 |      51.7 |     19.33 |    6.328 |
| [RetinaNet-resnext50 MLPERF](./retinanet-resnext50-mlperf/onnx/network_f16.yaml) | 299.6 G | 37.9 M |   -    |  35.8 %   |   ONNX    |  800x800  | [Kalray/retinanet-resnext50-mlperf](https://huggingface.co/Kalray/retinanet-resnext50-mlperf) |      19.5 |     50.96 |    5.838 |
| [SSD-mobileNetV1 MLPERF](./ssd-mobilenet-v1-mlperf/onnx/network_f16.yaml)        |  2.45 G |  6.7 M |   -    |     -     |   ONNX    |  300x300  | [Kalray/ssd-mobilenet-v1-mlperf](https://huggingface.co/Kalray/ssd-mobilenet-v1-mlperf)       |     509.1 |      1.96 |    1.240 |
| [SSD-mobileNetV2](./ssd-mobilenet-v2/onnx/network_f16.yaml)                      |  3.71 G | 16.1 M |   -    |     -     |   ONNX    |  300x300  | [Kalray/ssd-mobilenet-v2](https://huggingface.co/Kalray/ssd-mobilenet-v2)                     |     269.6 |      3.71 |    0.999 |
| [SSD-resnet34 MLPERF](./ssd-resnet34-mlperf/onnx/network_f16.yaml)               | 433.1 G | 20.0 M |   -    |     -     |   ONNX    | 1200x1200 | [Kalray/ssd-resnet34-mlperf](https://huggingface.co/Kalray/ssd-resnet34-mlperf)               |         - |         - |        - |
| [YOLOv3](./yolov3/onnx/network_f16.yaml)                                         | 66.12 G | 61.9 M |   -    |  55.3 %   |   ONNX    |  416x416  | [Kalray/yolov3](https://huggingface.co/Kalray/yolov3)                                         |      75.5 |     13.24 |    4.988 |
| [YOLOv3-Tiny](./yolov3-tiny/onnx/network_f16.yaml)                               |  5.58 G |  8.9 M |   -    |  33.1 %   |   ONNX    |  416x416  | [Kalray/yolov3-tiny](https://huggingface.co/Kalray/yolov3-tiny)                               |     513.6 |      1.94 |    2.868 |
| [YOLOv4](./yolov4/onnx/network_f16.yaml)                                         |  66.17G | 64.3 M |   -    |  57.3 %   |   ONNX    |  416x416  | [Kalray/yolov4](https://huggingface.co/Kalray/yolov4)                                         |         - |         - |        - |
| [YOLOv4-CSP-Mish](./yolov4-csp-mish/onnx/network_f16.yaml)                       | 114.7 G | 52.9 M |   -    |  55.0 %   |   ONNX    |  608x608  | [Kalray/yolov4-csp-mish](https://huggingface.co/Kalray/yolov4-csp-mish)                       |      42.6 |     23.50 |    4.708 |
| [YOLOv4-CSP-Relu](./yolov4-csp-relu/onnx/network_f16.yaml)                       | 109.2 G | 52.9 M |   -    |  54.0 %   |   ONNX    |  608x608  | [Kalray/yolov4-csp-relu](https://huggingface.co/Kalray/yolov4-csp-relu)                       |      47.9 |     20.91 |    5.211 |
| [YOLOv4-CSP-S-Mish](./yolov4-csp-s-mish/onnx/network_f16.yaml)                   | 21.64 G |  8.3 M |   -    |  44.6 %   |   ONNX    |  608x608  | [Kalray/yolov4-csp-s-mish](https://huggingface.co/Kalray/yolov4-csp-s-mish)                   |      98.0 |     10.20 |    1.938 |
| [YOLOv4-CSP-S-Relu](./yolov4-csp-s-relu/onnx/network_f16.yaml)                   | 19.16 G |  8.3 M |   -    |  42.6 %   |   ONNX    |  608x608  | [Kalray/yolov4-csp-s-relu](https://huggingface.co/Kalray/yolov4-csp-s-relu)                   |     121.1 |      8.25 |    2.297 |
| [YOLOv4-CSP-X-Relu](./yolov4-csp-x-relu/onnx/network_f16.yaml)                   | 166.6 G | 99.6 M |   -    |     -     |   ONNX    |  640x480  | [Kalray/yolov4-csp-x-relu](https://huggingface.co/Kalray/yolov4-csp-x-relu)                   |      36.8 |     27.17 |    6.146 |
| [YOLOv4-Tiny](./yolov4-tiny/onnx/network_f16.yaml)                               |  6.92 G |  6.1 M |   -    |  40.2 %   |   ONNX    |  416x416  | [Kalray/yolov4-tiny](https://huggingface.co/Kalray/yolov4-tiny)                               |     574.1 |      1.74 |    3.977 |
| [YOLOv5m6-Lite](./yolov5m6-relu/onnx/network_f16.yaml)                           | 52.45 G | 35.5 M |   -    |  62.9 %   |   ONNX    |  640x640  | [Kalray/yolov5m6-relu](https://huggingface.co/Kalray/yolov5m6-relu)                           |     120.8 |      8.28 |    6.355 |
| [YOLOv5s](./yolov5s/onnx/network_f16.yaml)                                       |  17.3 G |  7.2 M |   -    |  56.8 %   |   ONNX    |  640x640  | [Kalray/yolov5s](https://huggingface.co/Kalray/yolov5s)                                       |     219.6 |      4.58 |    3.683 |
| [YOLOv5s6-Lite](./yolov5s6-relu/onnx/network_f16.yaml)                           | 17.44 G | 12.6 M |   -    |  56.0 %   |   ONNX    |  640x640  | [Kalray/yolov5s6-relu](https://huggingface.co/Kalray/yolov5s6-relu)                           |     271.1 |      3.68 |    4.735 |
| [YOLOv7](./yolov7/onnx/network_f16.yaml)                                         | 107.8 G | 36.9 M |   -    |  51.4 %   |   ONNX    |  640x640  | [Kalray/yolov7](https://huggingface.co/Kalray/yolov7)                                         |      44.7 |     22.34 |    4.736 |
| [YOLOv7-Tiny](./yolov7-tiny/onnx/network_f16.yaml)                               |  13.7 G |  6.2 M |   -    |  38.7 %   |   ONNX    |  640x640  | [Kalray/yolov7-tiny](https://huggingface.co/Kalray/yolov7-tiny)                               |     150.8 |      6.62 |    2.078 |
| [YOLOv8l](./yolov8l/onnx/network_f16.yaml)                                       | 166.0 G | 43.6 M |   -    |  53.9 %   |   ONNX    |  640x640  | [Kalray/yolov8l](https://huggingface.co/Kalray/yolov8l)                                       |      48.9 |     20.44 |    8.126 |
| [YOLOv8m](./yolov8m/onnx/network_f16.yaml)                                       |  78.9 G | 25.9 M |   -    |  52.9 %   |   ONNX    |  640x640  | [Kalray/yolov8m](https://huggingface.co/Kalray/yolov8m)                                       |      96.6 |     10.34 |    7.688 |
| [YOLOv8n](./yolov8n/onnx/network_f16.yaml)                                       |   8.7 G |  3.2 M |   -    |  37.3 %   |   ONNX    |  640x640  | [Kalray/yolov8n](https://huggingface.co/Kalray/yolov8n)                                       |     326.9 |      3.05 |    2.920 |
| [YOLOv8n-ReLU](./yolov8n-relu/onnx/network_f16.yaml)                             |   8.7 G |  3.2 M |   -    |  36.9 %   |   ONNX    |  640x640  | [Kalray/yolov8n-relu](https://huggingface.co/Kalray/yolov8n-relu)                             |     459.6 |      2.16 |    4.036 |
| [YOLOv8n-ReLU-VGA](./yolov8n-relu-vga/onnx/network_f16.yaml)                     |   6.6 G |  3.2 M |   -    |  36.9 %   |   ONNX    |  640x480  | [Kalray/yolov8n-relu-vga](https://huggingface.co/Kalray/yolov8n-relu-vga)                     |     526.8 |      1.89 |    3.470 |
| [YOLOv8s](./yolov8s/onnx/network_f16.yaml)                                       |  28.6 G | 11.2 M |   -    |  44.9 %   |   ONNX    |  640x640  | [Kalray/yolov8s](https://huggingface.co/Kalray/yolov8s)                                       |     179.6 |      5.57 |    5.202 |
| [YOLOv8s-ReLU](./yolov8s-relu/onnx/network_f16.yaml)                             |  28.6 G | 11.2 M |   -    |  43.9 %   |   ONNX    |  640x640  | [Kalray/yolov8s-relu](https://huggingface.co/Kalray/yolov8s-relu)                             |     226.0 |      4.41 |    6.483 |
| [YOLOv9m](./yolov9m/onnx/network_f16.yaml)                                       |  76.3 G | 20.0 M |   -    |  51.4 %   |   ONNX    |  640x640  | [Kalray/yolov9m](https://huggingface.co/Kalray/yolov9m)                                       |      14.8 |     67.40 |    1.147 |
| [YOLOv9s](./yolov9t/onnx/network_f16.yaml)                                       |  26.4 G |  7.2 M |   -    |  46.8 %   |   ONNX    |  640x640  | [Kalray/yolov9s](https://huggingface.co/Kalray/yolov9s)                                       |      24.6 |     40.61 |    0.980 |
| [YOLOv9t](./yolov9t/onnx/network_f16.yaml)                                       |   7.7 G |  2.0 M |   -    |  38.3 %   |   ONNX    |  640x640  | [Kalray/yolov9t](https://huggingface.co/Kalray/yolov9t)                                       |      49.5 |     20.21 |    0.396 |
<!-- END AUTOMATED TABLE -->

