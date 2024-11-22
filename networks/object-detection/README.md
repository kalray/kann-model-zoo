<img width="20%" src="../../utils/materials/kalray_logo.png"></a>

## List of Object detection neural networks
This repository gives access to following object dectection Neural Networks main architecture:
* EfficientDet
* RetinatNet
* SSD
* YOLO

Please find the complete list here: </br>
![ModelsPass](https://img.shields.io/badge/ACE5.4-27%20pass-g)
![Models](https://img.shields.io/badge/Total%20model-27-blue)

<!-- START AUTOMATED TABLE -->
| NAME                                              |  FLOPs  | Params | mAP-50/95 | Framework   |  Dataset  | Input Size | ACE status                                            |
|:--------------------------------------------------|:-------:|:------:|:---------:|:------------|:---------:|:----------:|:------------------------------------------------------|
| [EfficientDet-D0](./efficientdet-d0)              | 10.2 G  | 3.9 M  |  53.0 %   | onnx        | COCO 2017 |  512x512   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [FasterRCNN-resnet50](./faster-rcnn-rn50)         | 94.1 G  | 26.8 M |  35.0 %   | onnx        | COCO 2017 |  512x512   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [RetinaNet-resnet50](./retinanet-resnet50)        | 122.4 G | 37.9 M |     -     | onnx        | COCO 2017 |  512x512   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [RetinaNet-resnet50](./retinanet-resnet50-mlperf) | 299.6 G | 37.9 M |  35.8 %   | onnx        | COCO 2017 |  800x800   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [RetinaNet-resnet101](./retinanet-resnet101)      | 161.4 G | 56.9 M |     -     | onnx        | COCO 2017 |  512x512   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [SSD-mobileNetV1](./ssd-mobilenet-v1-mlperf)      | 2.45 G  | 6.7 M  |     -     | tensorflow1 | COCO 2017 |  300x300   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [SSD-mobileNetV2](./ssd-mobilenet-v2)             | 3.71 G  | 16.1 M |     -     | tensorflow1 | COCO 2017 |  300x300   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [SSD-resnet34](./ssd-resnet34-mlperf)             | 433.1 G | 20.0 M |     -     | tensorflow1 | COCO 2017 | 1200x1200  | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv3](./yolov3)                                | 65.99 G | 61.9 M |  55.3 %   | tensorflow1 | COCO 2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv3](./yolov3)                                | 66.12 G | 61.9 M |  55.3 %   | onnx        | COCO 2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv3-Tiny](./yolov3-tiny)                      | 5.58 G  | 8.9 M  |  33.1 %   | tensorflow1 | COCO 2017 |  416x416   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv4-CSP-Mish](./yolov4-csp-mish)              | 114.7 G | 52.9 M |  55.0 %   | onnx        | COCO 2017 |  608x608   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv4-CSP-Relu](./yolov4-csp-relu)              | 109.2 G | 52.9 M |  54.0 %   | onnx        | COCO 2017 |  608x608   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv4-CSP-S-Mish](./yolov4-csp-s-mish)          | 21.64 G | 8.3 M  |  44.6 %   | onnx        | COCO 2017 |  608x608   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv4-CSP-S-Relu](./yolov4-csp-s-relu)          | 19.16 G | 8.3 M  |  42.6 %   | onnx        | COCO 2017 |  608x608   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv4-CSP-X-Relu](./yolov4-csp-x-relu)          | 166.6 G | 99.6 M |     -     | onnx        | COCO 2017 |  640x480   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv5s](./yolov5s)                              | 17.3 G  | 7.2 M  |  56.8 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv5m-Lite](./yolov5m6-relu)                   | 52.45 G | 35.5 M |  62.9 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv5s-Lite](./yolov5s6-relu)                   | 17.44 G | 12.6 M |  56.0 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv7](./yolov7)                                | 107.8 G | 36.9 M |  51.4 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv7-Tiny](./yolov7-tiny)                      | 13.7 G  | 6.2 M  |  38.7 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv8n](./yolov8n)                              |  8.7 G  | 3.2 M  |  37.3 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv8n-ReLU](./yolov8n-relu)                    |  8.7 G  | 3.2 M  |  36.9 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv8n-ReLU-VGA](./yolov8n-relu-vga)            |  6.6 G  | 3.2 M  |  36.9 %   | onnx        | COCO 2017 |  640x480   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv8s](./yolov8s)                              | 28.6 G  | 11.2 M |  44.9 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv8s-ReLU](./yolov8s-relu)                    | 28.6 G  | 11.2 M |  43.9 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv8m](./yolov8m)                              | 78.9 G  | 25.9 M |  52.9 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |
| [YOLOv8l](./yolov8l)                              | 166.0 G | 43.6 M |  53.9 %   | onnx        | COCO 2017 |  640x640   | ![Pass](https://img.shields.io/badge/ACE5.4-pass-g)   |

<!-- END AUTOMATED TABLE -->

![Ext](https://img.shields.io/badge/ACE5.4-ext-yellow) means that you have to use Custom layer
feature to run the model (and would be integrated in next ACE release). 

Please see [WIKI.md](../../WIKI.md) to use it.
