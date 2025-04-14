<img width="20%" src="../../utils/materials/kalray_logo.png"></a>

## List of Segmentation Neural Networks
This repository gives access to following segmentation neural networks main architecture:
* DeeplabV3+, Fully Convolution Network (FCN), U-Net, YOLO

## Important notes

* Neural networks are available on our **Hugging face plateform** 🤗 [HERE](https://huggingface.co/Kalray).
  Do not hesitate to check model cards for details of implementation, sources and/or license.

* Models have been trained on the following datasets: 
    + [COCO2017](https://cocodataset.org/#detection-2017)
    + PASCAL VOC-COCO 2017
    + MRI Brain
    + DAGM 2007

* To generate a neural network with KaNN :
    + in FP16, please refer to ONNX model (pointed by network_f16.yaml)
    + Please refer to [WIKI.md](../../WIKI.md) for instructions on how to use any of these models

## Neural Networks

The models are listed below, according the accuracy metrics (mAP50/95, mIoU) and MPPA performance in floating 
operations (FLOPS) per machine cycles (C) at batch 1.
The models are listed below, according:
  + the accuracy metrics (TopK accuracy)
  + MPPA performance at batch 1 in :
    * Frame per seconds (according that MPPA frequency is 1.0GHz along inference) from MPPA device
    * Total number of cycles to compute 1 frames (averaged on 10 first frames)
    * K Floating operations (FLOPS) per machine cycles (c)

*NB: MPPA Coolidge V2 processor default frequency is 1.0 GHz*

<!-- START AUTOMATED TABLE -->
| NAME                                                                            |   FLOPs | Params | mAP-50/95 |  mIoU  | Framework |  Input  | Dataset       | 🤗 HF repo-id                                                                                | FPS(MPPA) | TOTAL(Mc) | KFLOPS/c |
| :------------------------------------------------------------------------------ | ------: | -----: | :-------: | :----: | :-------: | :-----: | :------------ | ------------------------------------------------------------------------------------------- | --------: | --------: | -------: |
| [DeeplabV3Plus-mobilenet-V2](./deeplabv3plus-mobilenetv2/onnx/network_f16.yaml) |  17.4 G |  2.0 M |     -     |   -    |   ONNX    | 512x512 | VOC-COCO 2017 | [Kalray/deeplabv3plus-mobilenetv2](https://huggingface.co/Kalray/deeplabv3plus-mobilenetv2) |      86.3 |     11.56 |    1.486 |
| [DeeplabV3Plus-mobilenet-V3](./deeplabv3plus-mobilenetv3/onnx/network_f16.yaml) |  16.4 G |  8.9 M |     -     | 60.3 % |   ONNX    | 512x512 | VOC-COCO 2017 | []()                                                                                        |         - |         - |        - |
| [DeeplabV3Plus-Resnet50](./deeplabv3plus-resnet50/onnx/network_f16.yaml)        | 216.1 G | 39.6 M |     -     |   -    |   ONNX    | 416x416 | VOC-COCO 2017 | [Kalray/deeplabv3plus-resnet50](https://huggingface.co/Kalray/deeplabv3plus-resnet50)       |      27.6 |     36.19 |    1.806 |
| [FCN-Resnet101](./fcn-resnet101/onnx/network_f16.yaml)                          | 432.2 G | 51.8 M |     -     | 63.7 % |   ONNX    | 512x512 | VOC-COCO 2017 | [Kalray/fcn-resnet50](https://huggingface.co/Kalray/fcn-resnet50)                           |      15.5 |     64.37 |    6.730 |
| [FCN-Resnet50](./fcn-resnet50/onnx/network_f16.yaml)                            | 276.9 G | 32.9 M |     -     | 60.5 % |   ONNX    | 512x512 | VOC-COCO 2017 | [Kalray/fcn-resnet101](https://huggingface.co/Kalray/fcn-resnet101)                         |      21.8 |     45.95 |    6.047 |
| [UNet-2D-indus](./unet2d-tiny-ind/onnx/network_f16.yaml)                        |  36.7 G | 1.85 M |     -     |   -    |   ONNX    | 512x512 | DAGM-2007     | [Kalray/unet2d-tiny-ind](https://huggingface.co/Kalray/unet2d-tiny-ind)                     |     102.3 |      9.72 |    3.768 |
| [UNet-2D-medical](./unet2d-tiny-med/onnx/network_f16.yaml)                      |  24.4 G |  7.7 M |     -     |   -    |   ONNX    | 256x256 | MRI-BRAIN     | [Kalray/unet2d-tiny-med](https://huggingface.co/Kalray/unet2d-tiny-med)                     |     385.3 |      2.59 |    9.319 |
| [YOLOv8m-seg](./yolov8m-seg/onnx/network_f16.yaml)                              | 105.2 G | 27.2 M |  40.8 %   |   -    |   ONNX    | 640x640 | COCO 2017     | [Kalray/yolov8m-seg](https://huggingface.co/Kalray/yolov8m-seg)                             |      79.1 |     12.63 |    8.339 |
| [YOLOv8n-seg](./yolov8n-seg/onnx/network_f16.yaml)                              |  12.2 G |  3.4 M |  30.5 %   |   -    |   ONNX    | 640x640 | COCO 2017     | [Kalray/yolov8n-seg](https://huggingface.co/Kalray/yolov8n-seg)                             |     264.2 |      3.79 |    3.232 |
<!-- END AUTOMATED TABLE -->
