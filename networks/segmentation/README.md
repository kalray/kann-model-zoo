<img width="30%" src="https://upload.wikimedia.org/wikipedia/commons/4/46/Logo-KALRAY.png"></a>

## List of Segmentation Neural Networks

This repository gives access to following segmentation neural networks main architecture:

* DeeplabV3+, Fully Convolution Network (FCN), U-Net, YOLO

## Important notes

* Neural networks are available on our **Hugging face plateform** 🤗 [HERE](https://huggingface.co/Kalray).
  Do not hesitate to check model cards for details of implementation, sources and/or license.

* Models have been trained on the following datasets: 
  + [COCO2017](https://cocodataset.org/#detection-2017)
  + [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
  + [MRI Brain](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
  + [DAGM 2007](https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html)

* To generate a neural network with KaNN :
  + in FP16, refer to ONNX model (pointed by network_f16.yaml)
  + Please see [WIKI.md](../../WIKI.md) for instructions on how to use any of these models

  Example of use:
  ```bash
  # Generate
  kann generate ./networks/segmentation/unet2d-tiny-med/onnx/network_f16.yaml -d unet2d-med
  # wait ...
  # then, run
  kann run unet2d-med
  # observe the output to consider the global and detailed performance
  ```

## Neural Networks

The models are listed below, according:
  + The accuracy metrics (mAP50/95-mask or mIoU for segmentation)
  + MPPA performance at **batch 1** in :
    * Frames per second from MPPA device

*NB: MPPA Coolidge V2 processor default frequency is 1.0 GHz in ACE 6.0.0*

<!-- START AUTOMATED TABLE -->
| NAME                                                                            |   FLOPs | Params | mAP-50/95 |  mIoU  | Dtype |  Input  | Dataset               | 🤗 HF repo-id                                                                                | FPS (MPPA) |
| :------------------------------------------------------------------------------ | ------: | -----: | :-------: | :----: | :---: | :-----: | :-------------------- | ------------------------------------------------------------------------------------------- | --------: |
| [DeeplabV3Plus-mobilenet-V2](./deeplabv3plus-mobilenetv2/onnx/network_f16.yaml) |  17.4 G |  2.0 M |     -     | 55.3 % | FP16  | 512x512 | PASCAL-VOC            | [Kalray/deeplabv3plus-mobilenetv2](https://huggingface.co/Kalray/deeplabv3plus-mobilenetv2) |      86.3 |
| [DeeplabV3Plus-Resnet50](./deeplabv3plus-resnet50/onnx/network_f16.yaml)        | 216.1 G | 39.6 M |     -     | 60.8 % | FP16  | 416x416 | PASCAL-VOC            | [Kalray/deeplabv3plus-resnet50](https://huggingface.co/Kalray/deeplabv3plus-resnet50)       |      27.6 |
| [FCN-Resnet101](./fcn-resnet101/onnx/network_f16.yaml)                          | 432.2 G | 51.8 M |     -     | 63.7 % | FP16  | 512x512 | PASCAL-VOC / COCO2017 | [Kalray/fcn-resnet50](https://huggingface.co/Kalray/fcn-resnet50)                           |      15.5 |
| [FCN-Resnet50](./fcn-resnet50/onnx/network_f16.yaml)                            | 276.9 G | 32.9 M |     -     | 60.5 % | FP16  | 512x512 | PASCAL-VOC / COCO2017 | [Kalray/fcn-resnet101](https://huggingface.co/Kalray/fcn-resnet101)                         |      21.8 |
| [UNet-2D-indus](./unet2d-tiny-ind/onnx/network_f16.yaml)                        |  36.7 G | 1.85 M |     -     |   -    | FP16  | 512x512 | DAGM-2007             | [Kalray/unet2d-tiny-ind](https://huggingface.co/Kalray/unet2d-tiny-ind)                     |     102.3 |
| [UNet-2D-medical](./unet2d-tiny-med/onnx/network_f16.yaml)                      |  24.4 G |  7.7 M |     -     |   -    | FP16  | 256x256 | MRI-BRAIN             | [Kalray/unet2d-tiny-med](https://huggingface.co/Kalray/unet2d-tiny-med)                     |     385.3 |
| [YOLOv8m-seg](./yolov8m-seg/onnx/network_f16.yaml)                              | 105.2 G | 27.2 M |  40.8 %   |   -    | FP16  | 640x640 | COCO 2017             | [Kalray/yolov8m-seg](https://huggingface.co/Kalray/yolov8m-seg)                             |      79.1 |
| [YOLOv8n-seg](./yolov8n-seg/onnx/network_f16.yaml)                              |  12.2 G |  3.4 M |  30.5 %   |   -    | FP16  | 640x640 | COCO 2017             | [Kalray/yolov8n-seg](https://huggingface.co/Kalray/yolov8n-seg)                             |     264.2 |
<!-- END AUTOMATED TABLE -->
